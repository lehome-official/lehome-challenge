from pathlib import Path
import os
import argparse
import gc
from typing import Dict, Union, Set, Any, Optional, List, Tuple
import torch
from torch import Tensor
import gymnasium as gym
import cv2
import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.processor.core import TransitionKey
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lehome.utils.logger import get_logger

from .common import stabilize_garment_after_reset


logger = get_logger(__name__)


def convert_ee_pose_to_joints(
    ee_pose_action: torch.Tensor,
    current_joints: torch.Tensor,
    solver: Any,
    is_bimanual: bool,
    state_unit: str = "rad",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert end-effector pose action to joint angles using IK.

    Args:
        ee_pose_action: EE pose from policy. Single-arm: (8,), Bimanual: (16,)
        current_joints: Current joint positions for IK warm start. Single-arm: (6,), Bimanual: (12,)
        solver: RobotKinematics instance
        is_bimanual: Whether dual-arm or single-arm
        state_unit: 'rad' or 'deg'
        device: Target device for output tensor

    Returns:
        Joint angles tensor (same shape as current_joints)
    """
    from lehome.utils import compute_joints_from_ee_pose

    ee_pose_np = ee_pose_action.cpu().numpy()
    current_joints_np = current_joints.cpu().numpy()

    if is_bimanual:
        left_joints = compute_joints_from_ee_pose(
            solver,
            current_joints_np[:6],
            ee_pose_np[:8],
            state_unit,
            orientation_weight=0.01,
        )
        right_joints = compute_joints_from_ee_pose(
            solver,
            current_joints_np[6:12],
            ee_pose_np[8:16],
            state_unit,
            orientation_weight=0.01,
        )

        if left_joints is None:
            logger.warning("Left arm IK failed, using current joints")
            left_joints = current_joints_np[:6]
        if right_joints is None:
            logger.warning("Right arm IK failed, using current joints")
            right_joints = current_joints_np[6:12]

        joint_angles = np.concatenate([left_joints, right_joints])
    else:
        joint_angles = compute_joints_from_ee_pose(
            solver, current_joints_np, ee_pose_np, state_unit, orientation_weight=0.01
        )

        if joint_angles is None:
            logger.warning("IK failed, using current joints")
            joint_angles = current_joints_np

    return torch.from_numpy(joint_angles).float().to(device)


def preprocess_observation(
    obs_dict: Dict[str, Union[np.ndarray, Dict[str, Any]]],
    device: torch.device,
    task_description: str,
) -> Dict[str, Tensor]:
    """Preprocess observation dictionary into batched PyTorch tensors.

    Args:
        obs_dict: Observation dictionary from environment containing numpy arrays
        device: Target PyTorch device
        task_description: Task description string

    Returns:
        Dictionary with same structure, values as batched PyTorch tensors on device
    """
    processed_dict = {}
    for key, value in obs_dict.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            processed_dict[key] = preprocess_observation(
                value, device, task_description
            )
            continue

        # Assume the value is a numpy array from this point
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected numpy array for key '{key}', but got {type(value)}"
            )
        processed_value = value

        # Process image data: (H, W, C) -> (C, H, W), normalize to [0, 1]
        if processed_value.ndim == 3 and processed_value.shape[-1] == 3:
            assert (
                processed_value.dtype == np.uint8
            ), f"Image for key '{key}' expected np.uint8, got {processed_value.dtype}"
            processed_value = processed_value.astype(np.float32) / 255.0
            processed_value = np.transpose(processed_value, (2, 0, 1))

        batched_value = np.expand_dims(processed_value, axis=0)
        processed_dict[key] = torch.as_tensor(
            batched_value, dtype=torch.float32, device=device
        )
        processed_dict["task"] = task_description
    return processed_dict


def save_videos_from_observations(
    all_episode_frames: Dict[str, List[np.ndarray]],
    save_dir: str,
    episode_idx: int,
    success: torch.Tensor,
    fps: int = 30,
) -> None:
    if success.item():
        target_dir = os.path.join(save_dir, "success")
    else:
        target_dir = os.path.join(save_dir, "failure")

    os.makedirs(target_dir, exist_ok=True)

    for key, frames in all_episode_frames.items():
        if len(frames) == 0:
            continue
        h, w, c = frames[0].shape
        out_path = os.path.join(
            target_dir, f"episode{episode_idx}_{key.replace('.', '_')}.mp4"
        )
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()
        logger.info(f"Saved video: {out_path}")


def create_il_policy(policy_path: str, meta: Optional[LeRobotDatasetMetadata]) -> Any:
    """Create and return an IL (Imitation Learning) policy."""
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides={})
    policy_cfg.pretrained_path = policy_path

    # Auto-filter dataset metadata features to match policy requirements
    if meta is not None and hasattr(policy_cfg, "input_features"):
        expected_features = set(policy_cfg.input_features.keys())
        dataset_features = set(meta.features.keys())

        # Find features in dataset but not needed by policy
        extra_features = dataset_features - expected_features
        # Filter out system features (these are OK to have extra)
        system_features = {
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "next.done",
        }
        extra_features = extra_features - system_features

        # Remove extra observation features from metadata
        for feature in extra_features:
            if feature.startswith("observation."):
                logger.debug(f"Removing extra feature from metadata: {feature}")
                del meta.features[feature]

    policy = make_policy(policy_cfg, ds_meta=meta)
    policy.eval()
    policy.reset()
    return policy


def filter_observations_by_policy(
    obs_dict: Dict[str, Any], policy_input_features: Set[str]
) -> Dict[str, Any]:
    """Filter observation dictionary to only include features expected by policy.

    Args:
        obs_dict: Observation dictionary from environment
        policy_input_features: Set of feature keys expected by policy

    Returns:
        Filtered observation dictionary
    """
    filtered = {}
    for key, value in obs_dict.items():
        # Keep all non-observation keys (like internal env state)
        if not key.startswith("observation."):
            filtered[key] = value
        # Keep observation features that policy expects
        elif key in policy_input_features:
            filtered[key] = value
        # Silently skip observation features not needed by policy
    return filtered


def prepare_observation_for_preprocessor(
    observation_dict: Dict[str, Any],
    action_dim: int,
    task_description: str,
) -> Dict[str, Any]:
    """Prepare observation dictionary for preprocessor pipeline.

    Args:
        observation_dict: Raw observation dictionary from environment
        action_dim: Action dimension for dummy action tensor
        task_description: Task description string for VLA models

    Returns:
        Transition dictionary ready for preprocessor._forward()
    """
    obs_for_preproc = {}
    for key, value in observation_dict.items():
        if not key.startswith("observation."):
            continue

        if isinstance(value, np.ndarray):
            value_tensor = torch.from_numpy(value).float()
            if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                value_tensor = (
                    value_tensor.permute(2, 0, 1) / 255.0
                )  # (C, H, W), [0, 1]
                obs_for_preproc[key] = value_tensor.unsqueeze(0)  # (1, C, H, W)
            else:
                obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
        else:
            obs_for_preproc[key] = value

    if not isinstance(obs_for_preproc, dict):
        raise ValueError(f"obs_for_preproc must be a dict, got {type(obs_for_preproc)}")
    if len(obs_for_preproc) == 0:
        raise ValueError(
            "obs_for_preproc is empty after filtering. Check policy_input_features and observation_dict."
        )

    # Create transition format with complementary_data for VLA models
    dummy_action = torch.zeros(1, action_dim, dtype=torch.float32)
    transition = {
        TransitionKey.OBSERVATION: obs_for_preproc,
        TransitionKey.ACTION: dummy_action,
        TransitionKey.COMPLEMENTARY_DATA: {"task": task_description},
    }

    return transition


def process_observation(
    observation_dict: Dict[str, Any],
    preprocessor: Optional[Any],
    action_dim: int,
    task_description: str,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Process observation with preprocessor or manual preprocessing.

    Args:
        observation_dict: Raw observation dictionary from environment
        preprocessor: Optional preprocessor instance
        action_dim: Action dimension for dummy action tensor
        task_description: Task description string
        device: Target device for tensors

    Returns:
        Processed observation dictionary
    """
    if preprocessor is not None:
        transition = prepare_observation_for_preprocessor(
            observation_dict, action_dim, task_description
        )
        transformed_transition = preprocessor._forward(transition)
        return preprocessor.to_output(transformed_transition)
    else:
        return preprocess_observation(observation_dict, device, task_description)


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: Any,
    args: argparse.Namespace,
    meta: Optional[LeRobotDatasetMetadata] = None,
    preprocessor: Optional[Any] = None,
    postprocessor: Optional[Any] = None,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Core evaluation loop.

    Args:
        env: Environment instance
        policy: Policy instance
        args: Parsed command-line arguments
        meta: Dataset metadata
        preprocessor: Optional preprocessor for observations
        postprocessor: Optional postprocessor for actions
        ee_solver: Optional IK solver for end-effector pose conversion
        is_bimanual: Whether using dual-arm setup
        garment_name: Name of the garment being evaluated

    Returns:
        List of episode metrics dictionaries
    """
    # Get policy's expected input features for auto-filtering
    policy_input_features = (
        set(policy.config.input_features.keys())
        if hasattr(policy.config, "input_features")
        else None
    )

    # Get action dimension from meta or observation.state
    action_dim = None
    if meta is not None and hasattr(meta, "features") and "action" in meta.features:
        action_shape = meta.features["action"].get("shape", [])
        if isinstance(action_shape, list) and len(action_shape) > 0:
            action_dim = action_shape[0]
    # Fallback: infer from observation.state (usually same as action dim)
    if (
        action_dim is None
        and meta is not None
        and hasattr(meta, "features")
        and "observation.state" in meta.features
    ):
        state_shape = meta.features["observation.state"].get("shape", [])
        if isinstance(state_shape, list) and len(state_shape) > 0:
            action_dim = state_shape[0]
    # Final fallback: use default (12 for dual-arm, 6 for single-arm)
    if action_dim is None:
        # Check task name to determine if dual-arm
        if "Bi" in args.task or "bi" in args.task.lower():
            action_dim = 12  # Dual-arm
        else:
            action_dim = 6  # Single-arm

    logger.debug(f"Action dimension: {action_dim}")

    if args.save_datasets:
        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="abc",
            fps=30,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=meta.features if meta else None,
        )
        json_path = eval_dataset.root / "meta" / "garment_info.json"
        episode_index = 0
    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    for i in range(args.num_episodes):
        env.reset()
        policy.reset()

        # Stabilize garment after reset
        stabilize_garment_after_reset(env, args)

        # Get object initial pose after stabilization (if saving datasets)
        if args.save_datasets:
            object_initial_pose = env.get_all_pose()
        observation_dict = env._get_observations()
        # Auto-filter observations to match policy requirements
        if policy_input_features:
            observation_dict = filter_observations_by_policy(
                observation_dict, policy_input_features
            )

        # Initialize episode_frames for video saving
        if args.save_video:
            episode_frames = {k: [] for k in observation_dict.keys() if "images" in k}

        # Process observation with preprocessor
        observation = process_observation(
            observation_dict,
            preprocessor,
            action_dim,
            args.task_description,
            policy.config.device,
        )
        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)  # Initialize success tensor

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            with torch.inference_mode():
                action = policy.select_action(observation)

            # Apply postprocessor to denormalize actions
            if postprocessor is not None:
                action = postprocessor(action)
                if st == 0 and i == 0:
                    logger.debug(
                        f"Postprocessor applied - Action range: [{action.min().item():.3f}, {action.max().item():.3f}]"
                    )

            # Convert EE pose to joint angles if needed
            if args.use_ee_pose and ee_solver is not None:
                current_joints = observation_dict["observation.state"]
                if isinstance(current_joints, np.ndarray):
                    current_joints = (
                        torch.from_numpy(current_joints)
                        .float()
                        .to(policy.config.device)
                    )

                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=policy.config.device,
                ).unsqueeze(0)

                if st == 0 and i == 0:
                    logger.debug(
                        f"EE pose converted to joints - Range: [{action.min().item():.3f}, {action.max().item():.3f}]"
                    )

            env.step(action)
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50

            observation_dict = env._get_observations()
            # Auto-filter observations to match policy requirements
            if policy_input_features:
                observation_dict = filter_observations_by_policy(
                    observation_dict, policy_input_features
                )
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)

            # Process new observation
            observation = process_observation(
                observation_dict,
                preprocessor,
                action_dim,
                args.task_description,
                policy.config.device,
            )

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            reward = env.reward.item() if hasattr(env, "reward") else 0.0
            if not success_flag:
                episode_return += reward
                episode_length += 1

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # Get final success state before resetting
        is_success = success.item() if success_flag else False

        # Save episode data before resetting
        if args.save_datasets:
            if success_flag:
                eval_dataset.save_episode()
                append_episode_initial_pose(
                    json_path,
                    episode_index,
                    object_initial_pose,
                    garment_name=garment_name,
                )
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )
            episode_frames = {k: [] for k in observation_dict.keys() if "images" in k}

        # Record metrics
        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, "
            f"Length={episode_length}, Success={is_success}"
        )

    return all_episode_metrics


def calculate_and_print_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Calculate and print aggregated performance metrics.

    Args:
        metrics: List of episode metric dictionaries with 'return', 'length', 'success'
    """
    if not metrics:
        logger.info("[Results] No evaluation metrics were collected.")
        return

    total_returns = [m["return"] for m in metrics]
    total_successes = [1 if m["success"] else 0 for m in metrics]

    avg_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    success_rate = np.mean(total_successes)

    logger.info("=" * 50)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 50)
    logger.info(f"Total Episodes: {len(metrics)}")
    logger.info(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
    logger.info(f"Success Rate: {success_rate:.2%}")
    logger.info("=" * 50)


def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """Main evaluation function.

    Args:
        args: Parsed command-line arguments
        simulation_app: Isaac Sim application instance
    """
    # Parse environment configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)

    # Set random seed configuration
    if args.use_random_seed:
        env_cfg.use_random_seed = True
        logger.info("Using random seed")
    else:
        env_cfg.use_random_seed = False
        env_cfg.random_seed = args.seed
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        logger.info(f"Using fixed seed: {args.seed}")

    # Set garment configuration paths
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # Create the IL policy instance
    logger.info(f"Loading policy from: {args.policy_path}")
    meta = LeRobotDatasetMetadata(repo_id="abc", root=args.dataset_root)
    policy = create_il_policy(args.policy_path, meta)
    policy.to("cuda" if torch.cuda.is_available() else "cpu")

    # Create preprocessor and postprocessor
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    logger.info("Preprocessor and postprocessor created")

    # Initialize IK solver if using EE pose output
    ee_solver = None
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics

        urdf_path = Path(args.ee_urdf_path)
        if not urdf_path.is_absolute():
            urdf_path = Path.cwd() / urdf_path

        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        ee_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )
        logger.info(f"IK solver loaded ({'bimanual' if is_bimanual else 'single-arm'})")

    # Load eval list
    if args.stage == "single":
        eval_list = [args.garment_name]
        logger.info(f"Single mode: evaluating {args.garment_name}")
    else:
        # Capitalize first letter: release -> Release, holdout -> Holdout
        stage_capitalized = args.stage.capitalize()
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, stage_capitalized + "_list.txt"
        )
        with open(eval_list_path, "r") as f:
            eval_list = [line.strip() for line in f.readlines()]
        logger.info(f"Stage {args.stage}: loaded {len(eval_list)} garments")

    # Collect metrics for all garments
    all_garment_metrics: List[Dict[str, Any]] = []
    task_name = args.task

    logger.info("=" * 60)
    logger.info(f"Starting evaluation for {len(eval_list)} garments")
    logger.info("=" * 60)

    # Create environment with first garment
    switch_count = 0
    first_garment_name = eval_list[0] if eval_list else None
    if first_garment_name is None:
        logger.error("No garments found in eval list!")
        return

    env_cfg.garment_name = first_garment_name
    # Capitalize first letter: release -> Release, holdout -> Holdout
    env_cfg.garment_version = args.stage.capitalize()

    logger.info(f"Creating environment with garment: {first_garment_name}")
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    env.initialize_obs()
    logger.info("Environment initialized")

    try:
        # Evaluate each garment
        for garment_idx, garment_name in enumerate(eval_list):
            logger.info("=" * 60)
            logger.info(
                f"Evaluating garment {garment_idx + 1}/{len(eval_list)}: {garment_name}"
            )
            logger.info("=" * 60)

            # Switch to this garment (reuse the same environment)
            if garment_idx > 0:
                try:
                    if hasattr(env, "switch_garment"):
                        env.switch_garment(garment_name, args.stage.capitalize())
                        env.reset()
                        policy.reset()
                    else:
                        logger.warning(
                            "switch_garment not supported, recreating environment"
                        )
                        env.close()
                        env_cfg.garment_name = garment_name
                        env_cfg.garment_version = args.stage.capitalize()
                        env = gym.make(task_name, cfg=env_cfg).unwrapped
                        env.initialize_obs()
                        policy.reset()
                except Exception as e:
                    logger.error(f"Failed to switch garment: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

                switch_count += 1

            try:
                # Run the main evaluation loop for this garment
                episode_metrics = run_evaluation_loop(
                    env=env,
                    policy=policy,
                    args=args,
                    meta=meta,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    ee_solver=ee_solver,
                    is_bimanual=is_bimanual,
                    garment_name=garment_name,
                )

                # Store metrics with garment name
                all_garment_metrics.append(
                    {"garment_name": garment_name, "metrics": episode_metrics}
                )

                # Print metrics for this garment
                logger.info(f"Results for {garment_name}:")
                calculate_and_print_metrics(episode_metrics)

            except Exception as e:
                logger.error(f"Error evaluating garment {garment_name}: {e}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        logger.error(f"Critical error during evaluation: {e}")
        import traceback

        traceback.print_exc()

    # Print summary across all garments
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    if all_garment_metrics:
        # Aggregate all episode metrics
        all_episodes = []
        for garment_data in all_garment_metrics:
            for episode_metric in garment_data["metrics"]:
                episode_metric["garment_name"] = garment_data["garment_name"]
                all_episodes.append(episode_metric)

        # Print overall metrics
        calculate_and_print_metrics(all_episodes)

        # Print per-garment summary
        logger.info("=" * 60)
        logger.info("Per-Garment Summary")
        logger.info("=" * 60)
        for garment_data in all_garment_metrics:
            garment_name = garment_data["garment_name"]
            metrics = garment_data["metrics"]
            success_count = sum(1 for m in metrics if m["success"])
            success_rate = success_count / len(metrics) if metrics else 0.0
            avg_return = np.mean([m["return"] for m in metrics]) if metrics else 0.0
            logger.info(
                f"  {garment_name}: Success Rate = {success_rate:.2%}, Avg Return = {avg_return:.2f}"
            )
    else:
        logger.info("No metrics collected (all evaluations failed)")

    logger.info("=" * 60)
    logger.info("Starting cleanup...")
    logger.info("=" * 60)

    # Cleanup
    try:
        logger.info("Closing environment...")
        env.close()
    except Exception as e:
        logger.warning(f"Error closing environment: {e}")
        import traceback

        traceback.print_exc()
