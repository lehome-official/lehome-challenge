"""
BimanualIKSolver - Simple API for bimanual robot inverse kinematics.

This module provides a simplified interface for computing joint angles 
from world-space target positions, with fixed base poses for left and right arms.

Typical use case: Affordance detection or traditional grasping methods.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .kinematics import RobotKinematics
from .ee_pose_utils import compute_joints_from_world_point_detailed


class BimanualIKSolver:
    """
    Bimanual IK solver with fixed base poses.
    
    This class simplifies the workflow for bimanual robot control:
    1. Initialize once with URDF and fixed base poses
    2. Call solve_ik() repeatedly with different target positions
    
    Example:
        >>> solver = BimanualIKSolver(
        ...     urdf_path="Assets/robots/so101_new_calib.urdf",
        ...     left_base_pose=([1.15, -2.3, 0.5], [0.707, 0, 0, 0.707]),
        ...     right_base_pose=([1.65, -2.3, 0.5], [0.707, 0, 0, 0.707])
        ... )
        >>> 
        >>> # Solve IK for right arm to reach a target
        >>> target_pos = np.array([1.65, -2.03, 0.8])
        >>> joint_angles = solver.solve_ik(target_pos, arm='right')
        >>> if joint_angles is not None:
        ...     print(f"Joint angles: {joint_angles[:5]}")  # 5 arm joints
    """
    
    def __init__(
        self,
        urdf_path: str | Path,
        left_base_pose: Tuple[list, list],
        right_base_pose: Tuple[list, list],
        joint_names: Optional[list] = None,
        target_frame_name: str = "gripper_frame_link",
    ):
        """
        Initialize the bimanual IK solver.
        
        Args:
            urdf_path: Path to the URDF file (relative or absolute)
            left_base_pose: Tuple of (position, quaternion) for left arm base
                           - position: [x, y, z] in meters
                           - quaternion: [w, x, y, z]
            right_base_pose: Tuple of (position, quaternion) for right arm base
                            - position: [x, y, z] in meters
                            - quaternion: [w, x, y, z]
            joint_names: List of joint names to control (default: standard 5-joint arm)
            target_frame_name: Name of the end-effector frame in URDF
        
        Example:
            >>> left_pose = ([1.15, -2.3, 0.5], [0.707, 0, 0, 0.707])
            >>> right_pose = ([1.65, -2.3, 0.5], [0.707, 0, 0, 0.707])
            >>> solver = BimanualIKSolver("robot.urdf", left_pose, right_pose)
        """
        # Default joint names for SO101 robot
        if joint_names is None:
            joint_names = [
                "shoulder_pan",
                "shoulder_lift", 
                "elbow_flex",
                "wrist_flex",
                "wrist_roll"
            ]
        
        # Convert to Path object and resolve
        urdf_path = Path(urdf_path)
        if not urdf_path.is_absolute():
            # If relative, assume it's relative to the repository root
            # Try to find the repository root by looking for common markers
            current = Path.cwd()
            while current != current.parent:
                if (current / "Assets").exists() or (current / "source").exists():
                    urdf_path = current / urdf_path
                    break
                current = current.parent
        
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Create kinematics solver
        self.solver = RobotKinematics(
            str(urdf_path),
            target_frame_name=target_frame_name,
            joint_names=joint_names
        )
        
        # Store base poses
        self.left_base_pos = np.array(left_base_pose[0], dtype=np.float32)
        self.left_base_quat = np.array(left_base_pose[1], dtype=np.float32)  # [w,x,y,z]
        
        self.right_base_pos = np.array(right_base_pose[0], dtype=np.float32)
        self.right_base_quat = np.array(right_base_pose[1], dtype=np.float32)  # [w,x,y,z]
        
        # Store default initial joint configuration (typical working pose)
        self.default_initial_joints = np.array([0.0, -0.5, 0.8, 0.5, 0.0, 0.0])
        
        # Calculate actual separation between arms
        arm_separation = abs(self.right_base_pos[0] - self.left_base_pos[0])
        
        print(f"✓ BimanualIKSolver initialized")
        print(f"  URDF: {urdf_path}")
        print(f"  Left base:  pos={self.left_base_pos}, quat={self.left_base_quat}")
        print(f"  Right base: pos={self.right_base_pos}, quat={self.right_base_quat}")
        print(f"  Arm separation (X-axis): {arm_separation:.3f}m")
    
    def solve_ik(
        self,
        target_pos_world: np.ndarray | list,
        arm: str = "right",
        target_quat_world: Optional[np.ndarray | list] = None,
        initial_joints: Optional[np.ndarray] = None,
        state_unit: str = "rad",
        gripper_angle: float = 0.0,
        verbose: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for a target position in world coordinates.
        
        Args:
            target_pos_world: Target position in world frame [x, y, z] (meters)
            arm: Which arm to use, "left" or "right" (default: "right")
            target_quat_world: Optional target orientation [w, x, y, z]
                              If None, uses position-only IK (recommended for 5-DOF arms)
            initial_joints: Initial joint angles for IK warm start (6D array)
                           If None, uses default working pose
            state_unit: Joint angle unit, 'rad' or 'deg' (default: 'rad')
            gripper_angle: Gripper angle in same unit as joints (default: 0.0)
            verbose: Print detailed debug information (default: False)
        
        Returns:
            Joint angles [6D: 5 arm joints + 1 gripper] or None if IK fails
        
        Example:
            >>> # Simple position-only IK
            >>> target = [1.65, -2.03, 0.8]
            >>> joints = solver.solve_ik(target, arm='right')
            >>> 
            >>> # With orientation constraint
            >>> target_quat = [0, 0, 0, 1]  # No rotation
            >>> joints = solver.solve_ik(target, arm='right', target_quat_world=target_quat)
        """
        # Validate arm selection
        if arm not in ["left", "right"]:
            raise ValueError(f"Invalid arm '{arm}'. Must be 'left' or 'right'.")
        
        # Convert input to numpy arrays
        target_pos_world = np.array(target_pos_world, dtype=np.float32)
        if target_quat_world is not None:
            target_quat_world = np.array(target_quat_world, dtype=np.float32)
        
        # Select base pose based on arm
        if arm == "left":
            base_pos = self.left_base_pos
            base_quat = self.left_base_quat
        else:
            base_pos = self.right_base_pos
            base_quat = self.right_base_quat
        
        # Use default initial joints if not provided
        if initial_joints is None:
            initial_joints = self.default_initial_joints.copy()
        
        if verbose:
            print(f"\n[IK Solver] Solving for {arm} arm")
            print(f"  Target position (world): {target_pos_world}")
            print(f"  Base position (world): {base_pos}")
            print(f"  Base quaternion (world): {base_quat}")
        
        # Call the detailed IK function
        result = compute_joints_from_world_point_detailed(
            solver=self.solver,
            target_pos_world=target_pos_world,
            base_pos_world=base_pos,
            base_quat_world=base_quat,
            current_joints=initial_joints,
            state_unit=state_unit,
            target_quat_world=target_quat_world,
            gripper_angle=gripper_angle,
        )
        
        # Extract joint angles from result
        joint_angles = result["joints"]
        
        if verbose:
            if joint_angles is not None:
                print(f"  ✓ IK solved successfully")
                print(f"    Joint angles ({state_unit}): {joint_angles[:5]}")
                if "target_pos_base" in result and result["target_pos_base"] is not None:
                    print(f"    Target position (base frame): {result['target_pos_base']}")
            else:
                print(f"  ✗ IK failed: {result.get('error_msg', 'Unknown error')}")
        
        return joint_angles
    
    def get_base_pose(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the base pose for a specific arm.
        
        Args:
            arm: "left" or "right"
        
        Returns:
            Tuple of (position, quaternion) as numpy arrays
        
        Example:
            >>> pos, quat = solver.get_base_pose('right')
            >>> print(f"Right arm base: pos={pos}, quat={quat}")
        """
        if arm == "left":
            return self.left_base_pos.copy(), self.left_base_quat.copy()
        elif arm == "right":
            return self.right_base_pos.copy(), self.right_base_quat.copy()
        else:
            raise ValueError(f"Invalid arm '{arm}'. Must be 'left' or 'right'.")
    
    def set_default_initial_joints(self, joints: np.ndarray | list):
        """
        Set default initial joint configuration for IK warm start.
        
        Args:
            joints: 6D array of joint angles [5 arm joints + 1 gripper]
        
        Example:
            >>> solver.set_default_initial_joints([0.0, -0.5, 0.8, 0.5, 0.0, 0.0])
        """
        self.default_initial_joints = np.array(joints, dtype=np.float32)
        print(f"Default initial joints updated: {self.default_initial_joints[:5]}")


# Convenience function for quick one-off IK solving
def solve_bimanual_ik_simple(
    urdf_path: str | Path,
    target_pos_world: np.ndarray | list,
    arm: str = "right",
    left_base_pose: Tuple[list, list] = ([1.15, -2.3, 0.5], [0.707, 0, 0, 0.707]),
    right_base_pose: Tuple[list, list] = ([1.65, -2.3, 0.5], [0.707, 0, 0, 0.707]),
    **kwargs
) -> Optional[np.ndarray]:
    """
    One-line convenience function for IK solving (creates solver internally).
    
    For repeated calls, prefer creating a BimanualIKSolver instance instead.
    
    Args:
        urdf_path: Path to URDF file
        target_pos_world: Target position [x, y, z]
        arm: "left" or "right"
        left_base_pose: (position, quaternion) for left arm
        right_base_pose: (position, quaternion) for right arm
        **kwargs: Additional arguments passed to solve_ik()
    
    Returns:
        Joint angles or None if IK fails
    
    Example:
        >>> from lehome.utils.bimanual_ik_solver import solve_bimanual_ik_simple
        >>> joints = solve_bimanual_ik_simple(
        ...     "Assets/robots/so101_new_calib.urdf",
        ...     [1.65, -2.03, 0.8],
        ...     arm='right'
        ... )
    """
    solver = BimanualIKSolver(urdf_path, left_base_pose, right_base_pose)
    return solver.solve_ik(target_pos_world, arm=arm, **kwargs)

