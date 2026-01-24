import os
import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)

from lehome.utils.record import RateLimiter, get_next_experiment_path_with_gap, append_episode_initial_pose
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .common import stabilize_garment_after_reset
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: BasePolicy,  
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Core evaluation loop.
    Refactored to be agnostic of specific model implementations.
    """
    
    # --- Dataset Recording Setup (Optional) ---
    eval_dataset = None
    json_path = None
    episode_index = 0
    if args.save_datasets:
        # Note: We might need to handle 'features' argument if dataset saving is strictly required,
        # but for simplicity in Challenge mode, we often skip strictly matching metadata features.
        # Or you can expose policy.meta if available. 
        # Here we initialize vaguely to keep it simple.
        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="lehome_eval",
            fps=args.step_hz,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8, 
            image_writer_processes=0,
            features=None # Let LeRobot infer or pass explicitly if needed
        )
        json_path = eval_dataset.root / "meta" / "garment_info.json"

    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    for i in range(args.num_episodes):
        # 1. Reset Environment & Policy
        env.reset()
        policy.reset()
        stabilize_garment_after_reset(env, args)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()
        
        # Prepare for video recording
        episode_frames = {k: [] for k in observation_dict.keys() if "images" in k} if args.save_video else {}

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            # 3. Policy Inference (The core abstraction)
            # Input: Numpy Dict -> Output: Numpy Array
            action_np = policy.select_action(observation_dict)

            # 4. Prepare Action for Environment (Tensor)
            # Convert numpy action to tensor for Isaac Lab
            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

            # 5. Inverse Kinematics (Optional Helper Logic)
            # If policy outputs EE pose but env needs joints
            if args.use_ee_pose and ee_solver is not None:
                current_joints = torch.from_numpy(observation_dict["observation.state"]).float().to(args.device)
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)

            # 6. Step Environment
            env.step(action)
            
            # Check success
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50 # Run a bit longer after success to settle

            # Update Observation
            observation_dict = env._get_observations()

            # Recording
            if args.save_datasets:
                frame = {k: v for k, v in observation_dict.items() if k != "observation.top_depth"}
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            # Metrics
            reward = env.reward.item() if hasattr(env, "reward") else 0.0
            if not success_flag:
                episode_return += reward
                episode_length += 1

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False
        
        # Save Datasets
        if args.save_datasets:
            if success_flag:
                eval_dataset.save_episode()
                append_episode_initial_pose(json_path, episode_index, object_initial_pose, garment_name=garment_name)
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()

        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )

        # Log Metrics
        all_episode_metrics.append({"return": episode_return, "length": episode_length, "success": is_success})
        logger.info(f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}")

    return all_episode_metrics


def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """
    Main entry point for evaluation logic.
    """
    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    if args.use_random_seed:
        env_cfg.use_random_seed = True
    else:
        env_cfg.use_random_seed = False
        env_cfg.seed = args.seed
        # Propagate seed to sim config if structure exists
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed
            
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # 2. Initialize Policy (Using the Policy Registry)
    # This replaces create_il_policy, make_pre_post_processors, etc.
    logger.info(f"Initializing Policy Type: {args.policy_type}")
    
    # Check if policy is registered
    if not PolicyRegistry.is_registered(args.policy_type):
        available_policies = PolicyRegistry.list_policies()
        raise ValueError(
            f"Policy type '{args.policy_type}' not found in registry. "
            f"Available policies: {', '.join(available_policies)}"
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create policy instance from registry with appropriate arguments
    # Different policies may require different initialization arguments
    policy_kwargs = {"device": device}
    
    if args.policy_type == "lerobot":
        # LeRobot policy requires policy_path and dataset_root
        if not args.policy_path:
            raise ValueError("--policy_path is required for lerobot policy type")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for lerobot policy type")
        policy_kwargs.update({
            "policy_path": args.policy_path,
            "dataset_root": args.dataset_root,
            "task_description": args.task_description,
        })
    else:
        # Custom policy: participants can define their own loading logic
        # policy_path is optional and only passed if provided
        if args.policy_path:
            policy_kwargs.update({
                "model_path": args.policy_path,
            })
    
    # Create policy from registry
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    # 3. Initialize IK Solver (If needed)
    ee_solver = None
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics
        urdf_path = args.ee_urdf_path # Assuming path is handled or add check logic
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        ee_solver = RobotKinematics(str(urdf_path), target_frame_name="gripper_frame_link", joint_names=joint_names)
        logger.info(f"IK solver loaded.")

    # 4. Load Evaluation List
    stage_capitalized = args.stage.capitalize()  # Capitalize first letter: release -> Release, holdout -> Holdout
    if args.garment_type == "custom":
        # Default behavior: loads Release_test_list.txt
        list_filename = f"{stage_capitalized}_test_list.txt"
    else:
        # Map argument to filename
        type_map = {
            "tops_long": "Tops_Long",
            "tops_short": "Tops_Short",
            "trousers_long": "Trousers_Long",
            "trousers_short": "Trousers_Short"
        }
        # Get filename prefix, fallback to Tops_Long if something unexpected happens
        file_prefix = type_map.get(args.garment_type, "Tops_Long")
        list_filename = f"{file_prefix}/{file_prefix}.txt"

    eval_list_path = os.path.join(
        args.garment_cfg_base_path,
        stage_capitalized,
        list_filename
    )
    logger.info(f"Loading evaluation list from: {eval_list_path}")
    
    with open(eval_list_path, "r") as f:
        eval_list = [line.strip() for line in f.readlines()]
    logger.info(f"Stage {args.stage}: loaded {len(eval_list)} garments")
        
    # 5. Main Evaluation Loops
    all_garment_metrics = []
    
    # Init Env with first garment
    env_cfg.garment_name = eval_list[0]
    env_cfg.garment_version = args.stage.capitalize()
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    try:
        for garment_idx, garment_name in enumerate(eval_list):
            logger.info(f"Evaluating: {garment_name} ({garment_idx+1}/{len(eval_list)})")
            
            # Switch Garment Logic
            if garment_idx > 0:
                if hasattr(env, "switch_garment"):
                    env.switch_garment(garment_name, args.stage.capitalize())
                    env.reset()
                    policy.reset()
                else:
                    env.close()
                    env_cfg.garment_name = garment_name
                    env = gym.make(args.task, cfg=env_cfg).unwrapped
                    env.initialize_obs()
                    policy.reset()

            # Run Loop
            metrics = run_evaluation_loop(
                env=env,
                policy=policy,
                args=args,
                ee_solver=ee_solver,
                is_bimanual=is_bimanual,
                garment_name=garment_name
            )
            
            all_garment_metrics.append({"garment_name": garment_name, "metrics": metrics})
            calculate_and_print_metrics(metrics)

    finally:
        env.close()

    # 6. Final Summary
    logger.info("Overall Summary")
    if all_garment_metrics:
        all_episodes = [m for g in all_garment_metrics for m in g["metrics"]]
        calculate_and_print_metrics(all_episodes)