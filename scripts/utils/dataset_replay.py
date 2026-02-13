"""Dataset replay utility functions for replaying recorded episodes."""

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.utils.record import get_next_experiment_path_with_gap, RateLimiter
from lehome.utils.logger import get_logger

from .common import stabilize_garment_after_reset

logger = get_logger(__name__)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for dataset replay.

    Args:
        args: Command-line arguments containing replay configuration.

    Raises:
        ValueError: If dataset path is invalid, dataset is empty, or episode
            range is invalid.
    """
    dataset_path = Path(args.dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"Dataset root does not exist: {args.dataset_root}")

    info_json = dataset_path / "meta" / "info.json"
    if not info_json.exists():
        raise ValueError(f"Dataset info.json not found: {info_json}")

    # Check if dataset is empty
    try:
        with open(info_json, "r") as f:
            info = json.load(f)
        total_episodes = info.get("total_episodes", 0)
        if total_episodes == 0:
            raise ValueError(
                f"Dataset is empty (total_episodes=0). "
                f"Please use a dataset with recorded episodes. "
                f"Dataset path: {args.dataset_root}"
            )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse info.json: {e}")

    if args.num_replays < 1:
        raise ValueError(f"num_replays must be >= 1, got {args.num_replays}")

    # Validate episode range
    if args.start_episode < 0:
        raise ValueError(f"start_episode must be >= 0, got {args.start_episode}")

    if args.end_episode is not None:
        if args.end_episode < 0:
            raise ValueError(f"end_episode must be >= 0, got {args.end_episode}")
        if args.end_episode <= args.start_episode:
            raise ValueError(
                f"end_episode ({args.end_episode}) must be > start_episode ({args.start_episode})"
            )


def load_dataset(dataset_root: str) -> LeRobotDataset:
    """Load the LeRobotDataset from the specified root directory.

    Args:
        dataset_root: Root directory path of the dataset to load.

    Returns:
        Loaded LeRobotDataset instance.

    Raises:
        FileNotFoundError: If dataset data directory is not found.
        ValueError: If dataset has no episodes.
    """
    logger.info(f"Loading dataset from: {dataset_root}")

    # Check if required directories exist
    dataset_path = Path(dataset_root)
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset data directory not found: {data_dir}. "
            f"This dataset may be incomplete or corrupted."
        )

    try:
        dataset = LeRobotDataset(repo_id="replay_source", root=dataset_root)
        logger.info(
            f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames"
        )

        # Double-check that dataset actually has episodes
        if dataset.num_episodes == 0:
            raise ValueError(
                f"Dataset has no episodes (num_episodes=0). "
                f"Please use a dataset with recorded episodes. "
                f"Dataset path: {dataset_root}"
            )

        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def get_garment_name_from_json(dataset_root: str) -> str:
    """
    Parse garment_info.json to retrieve the garment name (the top-level key).
    
    Args:
        dataset_root: Root directory of the dataset.
        
    Returns:
        The name of the garment (e.g., 'Top_Long_Seen_0').
    """
    pose_file = Path(dataset_root) / "meta" / "garment_info.json"
    
    if not pose_file.exists():
        # Fallback for older jsonl format if necessary, though user specified json
        raise FileNotFoundError(f"Garment info file not found: {pose_file}")

    try:
        with open(pose_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not data:
            raise ValueError(f"Garment info file is empty: {pose_file}")
            
        # Get the first key in the dictionary (e.g., "Top_Long_Seen_0")
        # Assuming the dataset contains one type of garment or the first one is the target.
        garment_name = list(data.keys())[0]
        return garment_name
        
    except Exception as e:
        logger.error(f"Failed to extract garment name from {pose_file}: {e}")
        raise


def load_initial_pose(
    dataset_root: str, episode_index: int
) -> Optional[Dict[str, Any]]:
    """Load the initial object pose for a given episode from garment_info.json.

    The pose file format is:
    {
      "Top_Long_Unseen_0": {
        "0": {
          "object_initial_pose": [x, y, z, roll, pitch, yaw],
          "scale": [...]
        }
      }
    }

    Args:
        dataset_root: Root directory of the dataset.
        episode_index: Index of the episode to load pose for.

    Returns:
        Dictionary in format {"Garment": [x, y, z, roll, pitch, yaw]} for use with
        env.set_all_pose(), or None if not found.
    """
    pose_file = Path(dataset_root) / "meta" / "garment_info.json"
    if not pose_file.exists():
        # Try old JSONL format for backward compatibility
        pose_file_jsonl = Path(dataset_root) / "meta" / "garment_info.jsonl"
        if pose_file_jsonl.exists():
            logger.warning(f"Found old JSONL format, please migrate to new JSON format")
            return None
        logger.warning(f"Initial pose file not found: {pose_file}")
        return None

    try:
        with open(pose_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Search through all garments
        episode_key = str(episode_index)
        for garment_name, episodes in data.items():
            if episode_key in episodes:
                pose_list = episodes[episode_key].get("object_initial_pose")
                if pose_list is not None:
                    # Convert to format expected by set_all_pose: {"Garment": [...]}
                    return {"Garment": pose_list}

        logger.warning(f"Initial pose not found for episode {episode_index}")
        return None

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse {pose_file}: {e}")
        return None


def create_replay_dataset(
    args: argparse.Namespace, source_dataset: LeRobotDataset
) -> Tuple[Optional[LeRobotDataset], Optional[Path]]:
    """Create a new dataset for saving replayed episodes.

    Args:
        args: Command-line arguments containing output configuration.
        source_dataset: Source dataset to copy features from.

    Returns:
        Tuple of (replay_dataset, json_path):
            - replay_dataset: LeRobotDataset instance or None if output_root is None
            - json_path: Path to garment_info.json or None
    """
    if args.output_root is None:
        return None, None

    output_path = Path(args.output_root)
    source_folder_name = Path(args.dataset_root).name  # 获取 "005"
    root = output_path / source_folder_name
    if root.exists():
        logger.warning(f"Target path {root} already exists. Deleting it to create a fresh replay dataset.")
        shutil.rmtree(root)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Use the same features as the source dataset
    features = source_dataset.meta.features

    # Optionally remove depth if disabled
    if args.disable_depth and "observation.top_depth" in features:
        features = {k: v for k, v in features.items() if k != "observation.top_depth"}

    logger.info(f"Creating replay dataset at: {root}")
    replay_dataset = LeRobotDataset.create(
        repo_id="replay_output",
        fps=source_dataset.fps,
        root=root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )

    json_path = replay_dataset.root / "meta" / "garment_info.json"
    return replay_dataset, json_path


def compute_action_from_ee_pose(
    env: DirectRLEnv,
    frame_data: Dict[str, torch.Tensor],
    ik_solver: Any,
    is_bimanual: bool,
    args: argparse.Namespace,
    ik_stats: Dict[str, Any],
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """Compute joint angles from action.ee_pose using inverse kinematics.

    Args:
        env: Environment instance.
        frame_data: Frame data containing action.ee_pose and observation.state.
        ik_solver: Inverse kinematics solver instance.
        is_bimanual: Whether using dual-arm configuration.
        args: Command-line arguments containing ee_state_unit.
        ik_stats: Dictionary to track IK statistics (modified in-place).
        device: Device to place tensor on.

    Returns:
        Joint angle action tensor, or None if IK fails.
    """
    import torch
    from lehome.utils import compute_joints_from_ee_pose

    try:
        # Check if action.ee_pose exists
        if "action.ee_pose" not in frame_data:
            logger.warning(
                "action.ee_pose not found in frame data, falling back to original action"
            )
            ik_stats["total"] += 1
            ik_stats["fallback"] += 1
            return None

        # Get action.ee_pose from frame data
        action_ee_pose = frame_data["action.ee_pose"].cpu().numpy()

        # Use observation.state from dataset as IK initial guess
        # This represents the actual state before executing this action
        current_state = frame_data["observation.state"].cpu().numpy().flatten()

        if is_bimanual:
            # Dual-arm: solve IK separately for each arm
            left_ee = action_ee_pose[:8]
            right_ee = action_ee_pose[8:16]
            current_left = current_state[:6]
            current_right = current_state[6:12]

            # Left arm IK
            left_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_left,
                left_ee,
                args.ee_state_unit,
                orientation_weight=1.0,
            )
            # Right arm IK
            right_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_right,
                right_ee,
                args.ee_state_unit,
                orientation_weight=1.0,
            )

            if left_joints is None or right_joints is None:
                ik_stats["fallback"] += 1
                return None

            action_joints = np.concatenate([left_joints, right_joints], axis=0)
        else:
            # Single-arm IK
            action_joints = compute_joints_from_ee_pose(
                ik_solver,
                current_state,
                action_ee_pose,
                args.ee_state_unit,
                orientation_weight=1.0,
            )

            if action_joints is None:
                ik_stats["fallback"] += 1
                return None

        # Convert to tensor
        action_tensor = torch.from_numpy(action_joints).float().to(device).unsqueeze(0)

        # Record statistics
        ik_stats["total"] += 1
        ik_stats["success"] += 1

        # Compute error compared to original action
        original_action = frame_data["action"].cpu().numpy()
        error = np.max(np.abs(action_joints - original_action))
        ik_stats["errors"].append(error)

        return action_tensor

    except Exception as e:
        logger.warning(f"IK computation failed: {e}", exc_info=True)
        ik_stats["total"] += 1
        ik_stats["fallback"] += 1
        return None


def replay_episode(
    env: DirectRLEnv,
    episode_data: Any,
    rate_limiter: Optional[RateLimiter],
    initial_pose: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    replay_dataset: Optional[LeRobotDataset] = None,
    disable_depth: bool = False,
    ik_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    ik_stats: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    task_description: str = "fold the garment on the table",
) -> bool:
    """Replay a single episode from recorded data.

    Args:
        env: Environment instance.
        episode_data: Filtered dataset containing episode frames.
        rate_limiter: Rate limiter for controlling step frequency.
        initial_pose: Initial object pose dictionary.
        args: Command-line arguments.
        replay_dataset: Optional dataset for saving replayed observations.
        disable_depth: Whether to disable depth observation.
        ik_solver: Optional IK solver for ee_pose control.
        is_bimanual: Whether using dual-arm configuration.
        ik_stats: Optional dictionary to track IK statistics.
        device: Device to place tensors on.
        task_description: Task description string.

    Returns:
        True if episode was successful, False otherwise.
    """
    try:
        # Reset environment
        env.reset()

        # Set initial pose from recorded data (critical for reproducibility)
        # This ensures garment starts at the same position as during recording
        if initial_pose is not None:
            env.set_all_pose(initial_pose)
            logger.debug(f"Set initial pose from recorded data: {initial_pose}")
        else:
            logger.warning("No initial pose found in recorded data, using default pose")

        stabilize_garment_after_reset(env, args)

        success_achieved = False

        # Replay each frame
        for idx in range(len(episode_data)):
            if rate_limiter:
                rate_limiter.sleep(env)

            # Get action from recorded data
            if args.use_ee_pose and ik_solver is not None:
                # Use action.ee_pose + IK control
                action = compute_action_from_ee_pose(
                    env,
                    episode_data[idx],
                    ik_solver,
                    is_bimanual,
                    args,
                    ik_stats,
                    device,
                )
                if action is None:
                    # IK failed, fallback to original action
                    action = episode_data[idx]["action"].to(device).unsqueeze(0)
            else:
                # Directly use action (joint angles)
                action = episode_data[idx]["action"].to(device).unsqueeze(0)

            # Step environment
            env.step(action)

            # If saving, record observations
            if replay_dataset is not None:
                observations = env._get_observations()

                # Remove depth if disabled
                if disable_depth and "observation.top_depth" in observations:
                    observations = {
                        k: v
                        for k, v in observations.items()
                        if k != "observation.top_depth"
                    }
                frame = {**observations, "task": task_description}
                replay_dataset.add_frame(frame)

            # Check for success
            success = env._get_success().item()
            if success:
                success_achieved = True

        return success_achieved
    except Exception as e:
        logger.error(f"Error during episode replay: {e}", exc_info=True)
        return False


def append_episode_initial_pose(
    json_path: Path,
    episode_idx: int,
    object_initial_pose: Dict[str, Any],
    garment_name: Optional[str] = None,
    scale: Optional[Any] = None,
) -> None:
    """Append initial pose information to the JSON file.

    The JSON file format is:
    {
      "Top_Long_Unseen_0": {
        "0": {
          "object_initial_pose": [...],
          "scale": [...]
        }
      }
    }

    Args:
        json_path: Path to garment_info.json file.
        episode_idx: Episode index to save.
        object_initial_pose: Dictionary containing object initial pose.
        garment_name: Optional garment name.
        scale: Optional scale information.
    """
    from lehome.utils.record import append_episode_initial_pose as append_pose

    append_pose(
        json_path,
        episode_idx,
        object_initial_pose,
        garment_name=garment_name,
        scale=scale,
    )


def replay(args: argparse.Namespace) -> None:
    """Replay recorded datasets for visualization and verification.

    Args:
        args: Command-line arguments containing replay configuration.
    """
    validate_args(args)
    dataset = load_dataset(args.dataset_root)

    # Get device configuration (default to "cpu" for compatibility)
    device = getattr(args, "device", "cpu")
    task_description = getattr(
        args, "task_description", "fold the garment on the table"
    )

    ik_solver: Optional[Any] = None
    is_bimanual = False
    ik_stats: Dict[str, Any] = {"total": 0, "success": 0, "fallback": 0, "errors": []}

    if args.use_ee_pose:
        # Check if dataset contains ee_pose
        if "observation.ee_pose" not in dataset.meta.features:
            raise ValueError(
                "Dataset does not contain ee_pose. "
                "Please run augment_ee_pose.py or record with --record_ee_pose first."
            )

        urdf_path = Path(args.ee_urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        from lehome.utils import RobotKinematics

        # Determine single-arm or dual-arm configuration
        state_dim = dataset.meta.features["observation.state"]["shape"][0]
        is_bimanual = state_dim == 12

        # Get joint names (only first 5 joints, excluding gripper)
        joint_names = dataset.meta.features["observation.state"]["names"]
        if is_bimanual:
            solver_names = [n.replace("left_", "") for n in joint_names[:5]]
        else:
            solver_names = joint_names[:5]

        ik_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=solver_names,
        )
        arm_mode = "dual-arm" if is_bimanual else "single-arm"
        logger.info(f"IK solver loaded ({arm_mode} mode)")
        logger.warning(
            "Using action.ee_pose + IK control, which may differ from original action"
        )

    logger.info(f"Creating environment: {args.task}")
    env_cfg = parse_env_cfg(args.task, device=device)
    env_cfg.sim.use_fabric = False

    # Set garment configuration
    try:
        detected_garment_name = get_garment_name_from_json(args.dataset_root)
        logger.info(f"Auto-detected garment name from json: {detected_garment_name}")
        env_cfg.garment_name = detected_garment_name
    except Exception as e:
        logger.error(f"Could not determine garment name from dataset: {e}")
        raise
    env_cfg.garment_version = args.garment_version
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    env: DirectRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped

    # Initialize observations first (required for garment initialization)
    try:
        logger.info("Initializing observations...")
        env.initialize_obs()
        logger.info("Observations initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize observations: {e}", exc_info=True)
        env.close()
        raise

    # Create rate limiter
    rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None

    # Create replay dataset if output is specified
    replay_dataset, json_path = create_replay_dataset(args, dataset)

    # Determine episode range
    start_idx = args.start_episode
    end_idx = args.end_episode if args.end_episode is not None else dataset.num_episodes
    end_idx = min(end_idx, dataset.num_episodes)

    # Calculate total number of episodes for display (1-indexed)
    total_episodes = end_idx - start_idx

    logger.info(
        f"Replaying episodes {start_idx} to {end_idx - 1} (displayed as 1 to {total_episodes})"
    )

    # Statistics
    total_attempts = 0
    total_successes = 0
    saved_episodes = 0

    try:
        for episode_idx in range(start_idx, end_idx):
            # Display episode number starting from 1
            display_episode_num = episode_idx - start_idx + 1

            logger.info("")
            logger.info(f"{'=' * 60}")
            logger.info(f"Episode {display_episode_num}/{total_episodes}")
            logger.info(f"{'=' * 60}")

            # Load initial pose
            initial_pose = load_initial_pose(args.dataset_root, episode_idx)

            # Filter episode data
            try:
                episode_data = dataset.hf_dataset.filter(
                    lambda x: x["episode_index"].item() == episode_idx
                )
            except Exception as e:
                logger.error(
                    f"Failed to filter episode {display_episode_num} (index {episode_idx}) data: {e}"
                )
                continue

            if len(episode_data) == 0:
                logger.warning(
                    f"Episode {display_episode_num} (index {episode_idx}) has no data, skipping..."
                )
                continue

            logger.info(f"Episode length: {len(episode_data)} frames")

            # Replay multiple times if requested
            for replay_idx in range(args.num_replays):
                total_attempts += 1

                # Clear buffer if saving
                if replay_dataset is not None:
                    replay_dataset.clear_episode_buffer()

                # Replay the episode
                success = replay_episode(
                    env=env,
                    episode_data=episode_data,
                    rate_limiter=rate_limiter,
                    initial_pose=initial_pose,
                    args=args,
                    replay_dataset=replay_dataset,
                    disable_depth=args.disable_depth,
                    ik_solver=ik_solver,
                    is_bimanual=is_bimanual,
                    ik_stats=ik_stats,
                    device=device,
                    task_description=task_description,
                )

                if success:
                    total_successes += 1
                    logger.info(
                        f"  [Replay {replay_idx + 1}/{args.num_replays}] Success"
                    )
                else:
                    logger.info(
                        f"  [Replay {replay_idx + 1}/{args.num_replays}] Failed"
                    )

                # Save episode if conditions are met
                should_save = replay_dataset is not None and (
                    not args.save_successful_only or success
                )

                if should_save:
                    try:
                        replay_dataset.save_episode()
                        append_episode_initial_pose(
                            json_path, saved_episodes, initial_pose
                        )
                        saved_episodes += 1
                        logger.info(f"  Saved as episode {saved_episodes - 1}")
                    except Exception as e:
                        logger.error(f"Failed to save episode: {e}", exc_info=True)
                elif replay_dataset is not None:
                    replay_dataset.clear_episode_buffer()

    finally:
        # Ensure dataset is finalized even if an error occurs
        if replay_dataset is not None:
            try:
                replay_dataset.clear_episode_buffer()
                replay_dataset.finalize()
            except Exception as e:
                logger.error(f"Error finalizing dataset: {e}", exc_info=True)

    # Print statistics
    logger.info("")
    logger.info(f"{'=' * 60}")
    logger.info("Replay Statistics")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total attempts: {total_attempts}")
    logger.info(f"  Total successes: {total_successes}")
    if total_attempts > 0:
        logger.info(f"  Success rate: {100.0 * total_successes / total_attempts:.1f}%")
    if replay_dataset is not None:
        logger.info(f"  Saved episodes: {saved_episodes}")

    # IK statistics (if using ee_pose control)
    if args.use_ee_pose and ik_stats["total"] > 0:
        logger.info("")
        logger.info("IK Statistics")
        logger.info(f"  Total IK attempts: {ik_stats['total']}")
        logger.info(f"  IK successes: {ik_stats['success']}")
        logger.info(f"  IK fallbacks: {ik_stats['fallback']}")
        if ik_stats["total"] > 0:
            logger.info(
                f"  IK success rate: {100.0 * ik_stats['success'] / ik_stats['total']:.1f}%"
            )
        if ik_stats["errors"]:
            errors = np.array(ik_stats["errors"])
            unit = "rad" if args.ee_state_unit == "rad" else "deg"
            logger.info(f"  Joint angle error vs original action ({unit}):")
            logger.info(f"    mean = {np.mean(errors):.6f}")
            logger.info(f"    max  = {np.max(errors):.6f}")

    logger.info(f"{'=' * 60}")

    # Cleanup
    env.close()
