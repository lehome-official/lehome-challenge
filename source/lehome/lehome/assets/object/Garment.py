from typing import Optional, Sequence, Union, List

import os
import torch
import numpy as np
import random
from omegaconf import DictConfig, ListConfig

import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem, SingleXFormPrim
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Vt

from lehome.utils.logger import get_logger

# Create logger at module level using unified logger utility
# Will use global log file name if set, otherwise use default
logger = get_logger(__name__)


class GarmentObject(SingleClothPrim):
    """
    GarmentObject class that wraps the Isaac Sim SingleCloth prim functionality.
    This class inherits from the Isaac Sim SingleClothPrim class and can be extended

    Configuration Priority:
        - garment_config (from JSON): Highest priority, garment-specific settings
        - objects_config.common (from YAML): Fallback, default settings
    """

    def __init__(
        self,
        prim_path: str,
        particle_config: DictConfig,
        garment_config: DictConfig,
        rng: Optional[np.random.RandomState] = None,
    ):
        """
        Initialize the GarmentObject with position, orientation, and configuration.

        Args:
            prim_path: Path to the prim in the stage.
            particle_config: Particle system configuration (YAML).
            garment_config: Garment-specific configuration (JSON).
            rng: Optional random number generator (RandomState). If None, uses global random module.
                 This allows for reproducible random initialization when a fixed seed is used.

            1. set pos and ori for garment object
            2. create physics material and visual material for garment object
            3. randomize physics material and visual material according to the config
            4. record flat state for the garment object
        """
        # -------- Parameters Configuration ---------#
        # Store random number generator (for reproducible initialization)
        self.rng = rng

        # usd prim path
        self.usd_prim_path = prim_path
        logger.debug(f"usd prim path: {self.usd_prim_path}")
        # usd name
        self.prim_name = prim_path.split("/")[-1]
        self.prim_parent_path = os.path.dirname(self.usd_prim_path)
        # mesh prim path which is contained in the usd asset
        self.mesh_prim_path = find_unique_string_name(
            self.usd_prim_path + "/mesh",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        # particle system path
        self.particle_system_path = find_unique_string_name(
            self.usd_prim_path + "/particle_system",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        # particle material path
        self.particle_material_path = find_unique_string_name(
            self.usd_prim_path + "/particle_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        # Validate and store configurations (strict validation ensures all required fields exist)
        self._validate_configs(particle_config, garment_config)

        # Store configurations (validation ensures these are safe to access)
        self.particle_config = particle_config
        self.garment_config = garment_config
        self.objects_config = particle_config.objects  # Validated in _validate_configs

        # Extract visual_usd_paths (validated in _validate_configs, may be empty list)
        # Filter out None/null values, if all are null, treat as empty list
        if "visual_usd_paths" in garment_config and garment_config.visual_usd_paths:
            # Filter out None/null values from the list
            filtered_paths = [
                path for path in garment_config.visual_usd_paths if path is not None
            ]
            self.visual_usd_paths = filtered_paths if filtered_paths else []
        else:
            self.visual_usd_paths = []

        # Extract check_points (optional, validated in _validate_configs if present)
        if "check_point" in garment_config and garment_config.check_point is not None:
            self.check_points = list[int](garment_config.check_point)
            logger.debug(
                f"[GarmentObject] Loaded {len(self.check_points)} check points"
            )
        else:
            self.check_points = []

        # Extract success_distance (optional, validated in _validate_configs if present)
        if (
            "success_distance" in garment_config
            and garment_config.success_distance is not None
        ):
            self.success_distance = list[int](garment_config.success_distance)
            logger.debug(
                f"[GarmentObject] Loaded {len(self.success_distance)} success_distance"
            )
        else:
            self.success_distance = []

        # world prim encapsulation
        self.world_prim = SingleXFormPrim(self.usd_prim_path)

        # -------- Loading Procedure ---------#
        # get initial state
        logger.debug(f"[GarmentObject] Initializing garment at {prim_path}")
        self.init_pos, self.init_ori, self.init_scale = self._get_initial_pose()
        logger.debug(
            f"[GarmentObject] Initial pose - pos: {self.init_pos}, scale: {self.init_scale}"
        )

        # Load USD asset as a reference
        usd_path = self._get_usd_path()
        logger.debug(f"[GarmentObject] Loading USD asset from: {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        # define particle system for garment
        self.particle_system = SingleParticleSystem(
            prim_path=self.particle_system_path,
            particle_system_enabled=self.objects_config.particle_system.get(
                "particle_system_enabled", None
            ),
            enable_ccd=self.objects_config.particle_system.get("enable_ccd", None),
            solver_position_iteration_count=self.objects_config.particle_system.get(
                "solver_position_iteration_count", None
            ),
            max_depenetration_velocity=self.objects_config.particle_system.get(
                "max_depenetration_velocity", None
            ),
            global_self_collision_enabled=self.objects_config.particle_system.get(
                "global_self_collision_enabled", None
            ),
            non_particle_collision_enabled=self.objects_config.particle_system.get(
                "non_particle_collision_enabled", None
            ),
            contact_offset=self.objects_config.particle_system.get(
                "contact_offset", None
            ),
            rest_offset=self.objects_config.particle_system.get("rest_offset", None),
            particle_contact_offset=self.objects_config.particle_system.get(
                "particle_contact_offset", None
            ),
            fluid_rest_offset=self.objects_config.particle_system.get(
                "fluid_rest_offset", None
            ),
            solid_rest_offset=self.objects_config.particle_system.get(
                "solid_rest_offset", None
            ),
            wind=self.objects_config.particle_system.get("wind", None),
            max_neighborhood=self.objects_config.particle_system.get(
                "max_neighborhood", None
            ),
            max_velocity=self.objects_config.particle_system.get("max_velocity", None),
        )
        # define particle material for garment
        self.particle_material = ParticleMaterial(
            prim_path=self.particle_material_path,
            adhesion=self.objects_config.particle_material.get("adhesion", None),
            adhesion_offset_scale=self.objects_config.particle_material.get(
                "adhesion_offset_scale", None
            ),
            cohesion=self.objects_config.particle_material.get("cohesion", None),
            particle_adhesion_scale=self.objects_config.particle_material.get(
                "particle_adhesion_scale", None
            ),
            particle_friction_scale=self.objects_config.particle_material.get(
                "particle_friction_scale", None
            ),
            drag=self.objects_config.particle_material.get("drag", None),
            lift=self.objects_config.particle_material.get("lift", None),
            friction=self.objects_config.particle_material.get("friction", None),
            damping=self.objects_config.particle_material.get("damping", None),
            gravity_scale=self.objects_config.particle_material.get(
                "gravity_scale", None
            ),
            viscosity=self.objects_config.particle_material.get("viscosity", None),
            vorticity_confinement=self.objects_config.particle_material.get(
                "vorticity_confinement", None
            ),
            surface_tension=self.objects_config.particle_material.get(
                "surface_tension", None
            ),
        )
        self.num_count = 0
        # add particle cloth attribute to garment
        super().__init__(
            name=self.usd_prim_path,
            scale=self.init_scale,
            prim_path=self.mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=self.objects_config.garment_config.get("particle_mass", None),
            self_collision=self.objects_config.garment_config.get(
                "self_collision", None
            ),
            self_collision_filter=self.objects_config.garment_config.get(
                "self_collision_filter", None
            ),
            stretch_stiffness=self.objects_config.garment_config.get(
                "stretch_stiffness", None
            ),
            bend_stiffness=self.objects_config.garment_config.get(
                "bend_stiffness", None
            ),
            shear_stiffness=self.objects_config.garment_config.get(
                "shear_stiffness", None
            ),
            spring_damping=self.objects_config.garment_config.get(
                "spring_damping", None
            ),
        )

        if self.visual_usd_paths:
            logger.debug(
                f"[GarmentObject] Applying {len(self.visual_usd_paths)} visual materials"
            )
            self._apply_visual_material(self.visual_usd_paths)
        else:
            logger.warning("[GarmentObject] No visual materials specified")

        self.set_world_pose(position=self.init_pos, orientation=self.init_ori)
        logger.debug("[GarmentObject] Garment object initialized successfully")

        # refresh visibility
        # set_single_prim_visible(self.mesh_prim_path, visible=False)
        # set_single_prim_visible(self.mesh_prim_path, visible=True)

    def _validate_configs(
        self, particle_config: DictConfig, garment_config: DictConfig
    ):
        if garment_config is None:
            raise ValueError("garment_config cannot be None")
        if particle_config is None:
            raise ValueError("particle_config cannot be None")

        if not garment_config.get("asset_path"):
            raise ValueError("garment_config must contain a valid 'asset_path'")

        list_fields = ["visual_usd_paths", "check_point", "success_distance"]
        for field in list_fields:
            val = garment_config.get(field)
            if val is not None and not isinstance(val, (list, ListConfig)):
                raise ValueError(f"garment_config.{field} must be a list")

        if "visual_usd_paths" not in garment_config:
            logger.warning(
                "[GarmentObject] 'visual_usd_paths' missing, defaulting to empty."
            )

        if "objects" not in particle_config:
            raise ValueError("particle_config must contain 'objects' key")

        objects_cfg = particle_config.objects
        required_sections = [
            "common",
            "particle_system",
            "particle_material",
            "garment_config",
        ]
        missing = [s for s in required_sections if s not in objects_cfg]
        if missing:
            raise ValueError(f"particle_config.objects missing sections: {missing}")

        hybrid_params = [
            ("initial_pos_range", 6),
            ("initial_rot_range", 6),
            ("soft_reset_pos_range", 6),
            ("soft_reset_rot_range", 6),
            ("scale", 3),
        ]

        for param, expected_len in hybrid_params:
            val = garment_config.get(param) or objects_cfg.common.get(param)
            if val is None:
                raise ValueError(
                    f"Parameter '{param}' missing in garment_config and objects.common"
                )
            if hasattr(val, "__len__") and len(val) != expected_len:
                raise ValueError(
                    f"Invalid length for '{param}': expected {expected_len}, got {len(val)}"
                )

        logger.debug("[GarmentObject] Configuration validation passed.")

    def _get_usd_path(self) -> str:
        """
        Get USD asset path from garment_config.

        Note: asset_path is validated in _validate_configs, so it's guaranteed to exist.

        Returns:
            str: USD asset file path (converted to absolute path if starts with /)
        """
        usd_path = self.garment_config.asset_path
        # Convert relative path starting with / to absolute path using os.getcwd()
        if usd_path.startswith("/"):
            usd_path = os.path.join(os.getcwd(), usd_path[1:])
        return usd_path

    def _get_config_value(
        self, field_name: str, default_source: str = "common"
    ) -> tuple:
        """
        Get configuration value with priority: garment_config > objects_config.

        Note: For hybrid parameters (initial_pos_range, scale, etc.), existence is validated
        in _validate_configs, so this method is guaranteed to return a non-None value.
        For other parameters, may return None if not found.

        Args:
            field_name: Name of the configuration field to retrieve
            default_source: Source path in objects_config (e.g., "common", "particle_system")

        Returns:
            tuple: (value, source_name) where source_name indicates where the value came from
        """
        # Try garment_config first (validated in _validate_configs)
        if (
            field_name in self.garment_config
            and self.garment_config[field_name] is not None
        ):
            value = self.garment_config[field_name]
            source = "garment_config"
        else:
            # Fallback to objects_config (structure validated in _validate_configs)
            if default_source == "common":
                value = self.objects_config.common.get(field_name, None)
            else:
                sub_config = self.objects_config.get(default_source, {})
                value = sub_config.get(field_name, None)
            source = f"objects_config.{default_source}"

        return value, source

    def initialize(self):
        """
        Initialize the object by setting its initial position and orientation,
        while also get initial info of particles that make up the object.
        """
        # set local pose for initialization (wait for the update of scene manager)
        self.set_world_pose(position=self.init_pos, orientation=self.init_ori)

        if "cuda" in self._device:
            self.physics_sim_view = SimulationManager.get_physics_sim_view()
            self._cloth_prim_view.initialize(self.physics_sim_view)

        self._get_initial_info()

        self._prim.GetAttribute("points").Set(
            Vt.Vec3fArray.FromNumpy(self._get_points_pose().detach().cpu().numpy())
        )

    def reset(self):
        """
        Perform soft reset by randomly modifying the object's position and orientation.
        Meanwhile, return back to the initial positions of all particles that make up the object.
        """
        logger.debug("[GarmentObject] Performing soft reset")
        # Reset Points Positions
        if self._device == "cpu":
            self._prim.GetAttribute("points").Set(
                Vt.Vec3fArray.FromNumpy(self.initial_points_positions)
            )
        else:
            self._cloth_prim_view.set_world_positions(self.initial_points_positions)

        # Get position range from configuration
        pos_reset_range, pos_source = self._get_config_value(
            "soft_reset_pos_range", "common"
        )
        logger.debug(f"[GarmentObject] Using soft_reset_pos_range from {pos_source}")

        # Check if position range is valid (min != max for at least one dimension)
        pos_range_valid = (
            pos_reset_range[0] != pos_reset_range[3]
            or pos_reset_range[1] != pos_reset_range[4]
            or pos_reset_range[2] != pos_reset_range[5]
        )
        if not pos_range_valid:
            # Only warn once per session to avoid spam
            if not hasattr(self, "_warned_zero_pos_range"):
                logger.warning(
                    f"[GarmentObject] WARNING: soft_reset_pos_range has zero range "
                    f"(min == max for all dimensions). Reset position will always be the same: "
                    f"[{pos_reset_range[0]}, {pos_reset_range[1]}, {pos_reset_range[2]}]"
                )
                self._warned_zero_pos_range = True

        # Get rotation range from configuration
        rot_reset_range, rot_source = self._get_config_value(
            "soft_reset_rot_range", "common"
        )
        logger.debug(f"[GarmentObject] Using soft_reset_rot_range from {rot_source}")

        # Check if rotation range is valid (min != max for at least one dimension)
        rot_range_valid = (
            rot_reset_range[0] != rot_reset_range[3]
            or rot_reset_range[1] != rot_reset_range[4]
            or rot_reset_range[2] != rot_reset_range[5]
        )
        if not rot_range_valid:
            # Only warn once per session to avoid spam
            if not hasattr(self, "_warned_zero_rot_range"):
                logger.warning(
                    f"[GarmentObject] WARNING: soft_reset_rot_range has zero range "
                    f"(min == max for all dimensions). Reset rotation will always be the same: "
                    f"[{rot_reset_range[0]}, {rot_reset_range[1]}, {rot_reset_range[2]}]"
                )
                self._warned_zero_rot_range = True

        # Generate random initial position/rotation within range
        # Use provided rng if available, otherwise use global random module
        if self.rng is not None:
            pos = [
                self.rng.uniform(pos_reset_range[0], pos_reset_range[3]),
                self.rng.uniform(pos_reset_range[1], pos_reset_range[4]),
                self.rng.uniform(pos_reset_range[2], pos_reset_range[5]),
            ]
            ori = [
                self.rng.uniform(rot_reset_range[0], rot_reset_range[3]),
                self.rng.uniform(rot_reset_range[1], rot_reset_range[4]),
                self.rng.uniform(rot_reset_range[2], rot_reset_range[5]),
            ]
            logger.debug(
                f"[GarmentObject] Using RandomState for reset (rng state: {self.rng.get_state()[1][:3] if hasattr(self.rng, 'get_state') else 'N/A'})"
            )
        else:
            pos = [
                random.uniform(pos_reset_range[0], pos_reset_range[3]),
                random.uniform(pos_reset_range[1], pos_reset_range[4]),
                random.uniform(pos_reset_range[2], pos_reset_range[5]),
            ]
            ori = [
                random.uniform(rot_reset_range[0], rot_reset_range[3]),
                random.uniform(rot_reset_range[1], rot_reset_range[4]),
                random.uniform(rot_reset_range[2], rot_reset_range[5]),
            ]
            logger.debug(f"[GarmentObject] Using global random module for reset")

        self.set_world_pose(pos, euler_angles_to_quat(ori, degrees=True))
        self.reset_pose = np.concatenate(
            [np.array(pos, dtype=np.float32), np.array(ori, dtype=np.float32)]
        )
        logger.info(f"[GarmentObject] Reset complete - pos: {pos}, ori: {ori}")

    def get_current_mesh_points(
        self, visualize=False, save=False, save_path="./pointcloud.ply"
    ):
        """
        Get the current mesh points of the garment.
        Input:
            visualize: whether to visualize the mesh points
            save: whether to save the mesh points
            save_path: the path to save the mesh points
        Output:
            transformed_points: the current transformed mesh points of the garment, which is used for actual visualization
            mesh_points: the current original mesh points of the garment
            pos_world: the current world position of the garment (This parameter is suitable for cpu version, which will be set to None in gpu version)
            ori_world: the current world orientation of the garment (This parameter is suitable for cpu version, which will be set to None in gpu version)
        """
        if self._device == "cpu":
            pos_world, ori_world = self.get_world_pose()
            scale_world = self.get_world_scale()
            mesh_points = self._get_points_pose().detach().cpu().numpy()
            transformed_mesh_points = self.transform_points(
                mesh_points,
                pos_world.detach().cpu().numpy(),
                ori_world.detach().cpu().numpy(),
                scale_world.detach().cpu().numpy(),
            )
        else:
            mesh_points = (
                self._cloth_prim_view.get_world_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            transformed_mesh_points = mesh_points
            pos_world = None
            ori_world = None
        # visualize the initial points
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_mesh_points)
        if visualize:
            o3d.visualization.draw_geometries([pcd])
        if save:
            o3d.io.write_point_cloud(save_path, pcd)
            logger.debug(f"points saved to {save_path}")
        return transformed_mesh_points, mesh_points, pos_world, ori_world

    def _apply_visual_material(self, material_paths: Sequence[str]):
        """
        Load provided material USDs and bind them to mesh sub-prims by suffix:
        mesh -> material_paths[0], mesh1 -> material_paths[1], mesh2 -> material_paths[2], etc.
        """
        loaded_materials = []

        # 1. Loop through and load all material USD files from the list
        for i, path in enumerate(material_paths):
            # Convert relative path starting with / to absolute path using os.getcwd()
            if path.startswith("/"):
                path = os.path.join(os.getcwd(), path[1:])

            # Create a unique Prim path for each material, e.g., /visual_material_0, /visual_material_1
            mat_prim_path = find_unique_string_name(
                f"{self.usd_prim_path}/visual_material_{i}",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )

            # Add the USD as a reference to the Stage
            add_reference_to_stage(usd_path=path, prim_path=mat_prim_path)

            # Resolve the actual Shader path used for binding
            vis_prim = prims_utils.get_prim_at_path(mat_prim_path)
            # Assume the material USD structure has a Shader/Material Prim under a parent node
            children = prims_utils.get_prim_children(vis_prim)
            if children:
                material_prim_resolved = children[0].GetPath()
                loaded_materials.append(material_prim_resolved)
            else:
                logger.warning(f"Warning: No material found in {path}")
                loaded_materials.append(None)

        # 2. Retrieve all child Prims under the Mesh
        self.mesh_prim = prims_utils.get_prim_at_path(self.mesh_prim_path)
        self.garment_submesh = prims_utils.get_prim_children(self.mesh_prim)

        # Case A: The mesh is a single Prim (no sub-meshes)
        if len(self.garment_submesh) == 0:
            if len(loaded_materials) > 0 and loaded_materials[0] is not None:
                # Bind the 0-th material directly to mesh_prim_path
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=self.mesh_prim_path,
                    material_path=loaded_materials[0],
                )

        # Case B: This is a group containing multiple sub-meshes (mesh, mesh1, mesh2...)
        else:
            for prim in self.garment_submesh:
                name = prim.GetName()
                target_idx = 0  # Default to the 0-th material

                # Name matching logic for material assignment
                if name == "mesh":
                    target_idx = 0
                elif name.startswith("mesh"):
                    try:
                        idx_str = name.replace("mesh", "")
                        target_idx = int(idx_str)
                    except ValueError:
                        target_idx = 0

                # Execute the material binding command
                if (
                    target_idx < len(loaded_materials)
                    and loaded_materials[target_idx] is not None
                ):
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=prim.GetPath(),
                        material_path=loaded_materials[target_idx],
                    )
                else:
                    # Fallback: skip or bind default if index is out of range (e.g., mesh4 but only 3 materials provided)
                    logger.warning(
                        f"[GarmentObject] Warning: Material index {target_idx} out of range for {name}"
                    )

    def _get_initial_pose(self):
        """
        Get the initial pose (/ori) of the garment object.
        Priority: garment_config > objects_config.common
        """
        # Get position range from configuration (prefer garment_config over objects_config)
        pos_init_range, pos_source = self._get_config_value(
            "initial_pos_range", "common"
        )
        logger.debug(f"[GarmentObject] Using initial_pos_range from {pos_source}")

        # Get rotation range from configuration (prefer garment_config over objects_config)
        rot_init_range, rot_source = self._get_config_value(
            "initial_rot_range", "common"
        )
        logger.debug(f"[GarmentObject] Using initial_rot_range from {rot_source}")

        # Generate random initial position/rotation within range
        # Use provided rng if available, otherwise use global random module
        if self.rng is not None:
            pos = [
                self.rng.uniform(pos_init_range[0], pos_init_range[3]),
                self.rng.uniform(pos_init_range[1], pos_init_range[4]),
                self.rng.uniform(pos_init_range[2], pos_init_range[5]),
            ]
            ori = [
                self.rng.uniform(rot_init_range[0], rot_init_range[3]),
                self.rng.uniform(rot_init_range[1], rot_init_range[4]),
                self.rng.uniform(rot_init_range[2], rot_init_range[5]),
            ]
        else:
            pos = [
                random.uniform(pos_init_range[0], pos_init_range[3]),
                random.uniform(pos_init_range[1], pos_init_range[4]),
                random.uniform(pos_init_range[2], pos_init_range[5]),
            ]
            ori = [
                random.uniform(rot_init_range[0], rot_init_range[3]),
                random.uniform(rot_init_range[1], rot_init_range[4]),
                random.uniform(rot_init_range[2], rot_init_range[5]),
            ]

        # Get scale from configuration (prefer garment_config over objects_config)
        scale, scale_source = self._get_config_value("scale", "common")
        logger.debug(f"[GarmentObject] Using scale from {scale_source}: {scale}")

        self.reset_pose = np.concatenate(
            [np.array(pos, dtype=np.float32), np.array(ori, dtype=np.float32)]
        )
        # Set initial pose
        return pos, euler_angles_to_quat(ori, degrees=True), scale

    def _get_initial_info(self):
        """
        Return the initial positions of all particles that make up the object.
        """
        if self._device == "cpu":
            self.initial_points_positions = (
                self._get_points_pose().detach().cpu().numpy()
            )
        else:
            self.initial_points_positions = self._cloth_prim_view.get_world_positions()

    def transform_points(self, points, pos, ori, scale):
        """
        Transform points by pos, ori and scale

        Args:
            points (numpy.ndarray): (N, 3) points to be transformed
            pos (numpy.ndarray): (3,) position transformation of the object
            ori (numpy.ndarray): (4,) orientation transformation of the object (quaternion)
            scale (int): scale transformation of the object
        """
        ori_matrix = quat_to_rot_matrix(ori)
        scaled_points = points * scale
        transformed_points = scaled_points @ ori_matrix.T + pos
        return transformed_points

    def inverse_transform_points(self, transformed_points, pos, ori, scale):
        """
        Inverse transform: Recover original points from transformed ones using pos, ori, and scale.

        Args:
            transformed_points (numpy.ndarray): (N, 3) transformed points in world space
            pos (numpy.ndarray): (3,) position transformation of the object
            ori (numpy.ndarray): (4,) orientation transformation of the object (quaternion, xyzw)
            scale (float): scale transformation of the object

        Returns:
            numpy.ndarray: (N, 3) original local-space points
        """
        ori_matrix = quat_to_rot_matrix(ori)
        shifted_points = transformed_points - pos
        rotated_points = shifted_points @ ori_matrix
        original_points = rotated_points / scale
        return original_points

    def get_all_pose(self):
        return {"Garment": self.reset_pose}

    def set_all_pose(self, pose_dict: dict):
        if "Garment" in pose_dict:
            pose = pose_dict["Garment"]
            pos = pose[:3]
            ori = pose[3:]
            self.set_world_pose(
                [0.0, 0.0, 0.0], euler_angles_to_quat([0.0, 0.0, 0.0], degrees=True)
            )
            if self._device == "cpu":
                self._prim.GetAttribute("points").Set(
                    Vt.Vec3fArray.FromNumpy(self.initial_points_positions)
                )
            else:
                self._cloth_prim_view.set_world_positions(self.initial_points_positions)
            self.set_world_pose(pos, euler_angles_to_quat(ori, degrees=True))
            self.reset_pose = np.array(pose, dtype=np.float32)

            logger.info(
                f"[GarmentObject] set_all_pose Reset complete - pos: {pos}, ori: {ori}"
            )
