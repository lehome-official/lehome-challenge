import argparse


def setup_record_parser(
    subparsers: argparse.ArgumentParser, parent_parsers: list[argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    """Setup parser for 'record' subcommand."""
    parser = subparsers.add_parser(
        "record",
        help="Record teleoperation data",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )

    # Teleoperation Parameters
    parser.add_argument(
        "--teleop_device",
        type=str,
        default="keyboard",
        choices=["keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
        help="Device for interacting with environment",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Port for the teleop device:so101leader, default is /dev/ttyACM0",
    )
    parser.add_argument(
        "--left_arm_port",
        type=str,
        default="/dev/ttyACM0",
        help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
    )
    parser.add_argument(
        "--right_arm_port",
        type=str,
        default="/dev/ttyACM1",
        help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="recalibrate SO101-Leader or Bi-SO101Leader",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
    )
    # Task Configuration
    parser.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task.",
    )
    parser.add_argument(
        "--garment_name",
        type=str,
        default="Top_Long_Unseen_0",
        help="Name of the garment.",
    )
    parser.add_argument(
        "--garment_version", type=str, default="Release", help="Version of the garment."
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )
    parser.add_argument(
        "--use_random_seed",
        action="store_true",
        default=False,
        help="Use random seed for the environment.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment."
    )
    parser.add_argument(
        "--log_success",
        action="store_true",
        default=False,
        help="Log success information.",
    )
    # Recording Parameters
    parser.add_argument(
        "--enable_record",
        action="store_true",
        default=False,
        help="Enable dataset recording function",
    )
    parser.add_argument(
        "--step_hz", type=int, default=120, help="Environment stepping rate in Hz."
    )
    parser.add_argument(
        "--num_episode",
        type=int,
        default=20,
        help="Maximum number of episodes to record",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/record",
        help="Root directory for saving recorded datasets (default: Datasets/record)",
    )
    parser.add_argument(
        "--disable_depth",
        action="store_true",
        default=False,
        help="Disable using top depth observation in env and dataset.",
    )
    parser.add_argument(
        "--enable_pointcloud",
        action="store_true",
        default=False,
        help="Whether to enable pointcloud observation in env and dataset.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help=" Description of the task to be performed.",
    )
    parser.add_argument(
        "--record_ee_pose",
        action="store_true",
        default=False,
        help="Record end-effector pose online (requires Pinocchio and scipy)",
    )
    parser.add_argument(
        "--ee_urdf_path",
        type=str,
        default=None,
        help="URDF file path (required only when using --record_ee_pose)",
    )
    parser.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint angle unit for kinematic solver (default: rad)",
    )

    return parser


def setup_replay_parser(
    subparsers: argparse.ArgumentParser, parent_parsers: list[argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    """Setup parser for 'replay' subcommand."""
    parser = subparsers.add_parser(
        "replay",
        help="Replay dataset",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task environment.",
    )
    parser.add_argument(
        "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/record/example/record_top_long_release_10/001",
        help="Root directory of the dataset to replay.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory to save replayed episodes (if None, replay only without saving).",
    )
    parser.add_argument(
        "--num_replays",
        type=int,
        default=1,
        help="Number of times to replay each episode.",
    )
    parser.add_argument(
        "--save_successful_only",
        action="store_true",
        default=False,
        help="Only save episodes that achieve success during replay.",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Starting episode index (inclusive).",
    )
    parser.add_argument(
        "--end_episode",
        type=int,
        default=None,
        help="Ending episode index (exclusive). If None, replay all episodes.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help="Description of the task to be performed.",
    )
    parser.add_argument(
        "--garment_version", type=str, default="Release", help="Version of the garment."
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )
    parser.add_argument(
        "--use_ee_pose",
        action="store_true",
        default=False,
        help="Use action.ee_pose (Cartesian space) control, converted to joint angles via IK.",
    )
    parser.add_argument(
        "--ee_urdf_path",
        type=str,
        default="Assets/robots/so101_new_calib.urdf",
        help="URDF file path (required when using --use_ee_pose).",
    )
    parser.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint angle unit for kinematic solver (default: rad).",
    )
    parser.add_argument(
        "--disable_depth",
        action="store_true",
        default=False,
        help="Disable depth observation during replay.",
    )

    return parser


def setup_inspect_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'inspect' subcommand."""
    parser = subparsers.add_parser("inspect", help="Inspect dataset metadata")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--show_frames", type=int, default=None, help="Display first N frames"
    )
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_read_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'read' subcommand."""
    parser = subparsers.add_parser("read", help="Read dataset states")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--num_frames", type=int, default=None, help="Number of frames to read"
    )
    parser.add_argument(
        "--episode", type=int, default=None, help="Specific episode index"
    )
    parser.add_argument("--output_csv", type=str, default=None, help="Export to CSV")
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_augment_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'augment' subcommand."""
    parser = subparsers.add_parser("augment", help="Add end-effector pose to dataset")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument("--urdf_path", type=str, required=True, help="URDF file path")
    parser.add_argument(
        "--state_unit",
        type=str,
        default="rad",
        choices=["rad", "deg"],
        help="Joint angle unit",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory (default: in-place)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing EE pose data"
    )
    return parser


def setup_merge_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'merge' subcommand."""
    parser = subparsers.add_parser("merge", help="Merge multiple datasets")
    parser.add_argument(
        "--source_roots",
        type=str,
        required=True,
        help="List of source dataset directories (as Python list string)",
    )
    parser.add_argument(
        "--output_root", type=str, required=True, help="Output dataset directory"
    )
    parser.add_argument(
        "--output_repo_id", type=str, default="merged_dataset", help="Repository ID"
    )
    parser.add_argument(
        "--merge_custom_meta",
        action="store_true",
        default=True,
        help="Merge custom meta files",
    )
    return parser


def setup_eval_parser() -> argparse.ArgumentParser:
    """Setup parser for evaluation script.

    Returns:
        The parser with evaluation arguments added.
    """
    parser = argparse.ArgumentParser(
        description="A script for evaluating policy in lehome manipulation environments."
    )

    # Core arguments
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Maximum number of steps per evaluation episode.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run for each garment.",
    )
    parser.add_argument(
        "--step_hz", type=int, default=120, help="Environment stepping rate in Hz."
    )
    # Evaluation parameters
    parser.add_argument(
        "--use_random_seed",
        action="store_true",
        default=False,
        help="Use random seed for the environment.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="release",
        choices=["release", "holdout", "all"],
        help="Evaluation stage. 'release' and 'holdout' load garment list from file. 'all' evaluates all the garments.",
    )
    parser.add_argument(
        "--garment_type",
        type=str,
        default="tops_long",
        choices=["tops_long", "tops_short", "trousers_long", "trousers_short", "custom"],
        help="Type of garments to evaluate.",
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path to the garment configuration files.",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path to the particle configuration file.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help="Task description for VLA models (used in complementary_data).",
    )

    # Record parameters
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="If set, save evaluation episodes as video.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="outputs/eval_videos",
        help="Directory to save evaluation videos.",
    )
    parser.add_argument(
        "--save_datasets",
        action="store_true",
        help="If set, save evaluation episodes dataset(only success).",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="Datasets/eval",
        help="Path to save evaluation datasets.",
    )

    # Policy arguments for Imitation Learning (IL)
    # Note: Available policy types are dynamically loaded from PolicyRegistry
    parser.add_argument(
        "--policy_type",
        type=str,
        default="lerobot",
        help=(
            "Type of policy to use. Available policies are registered in PolicyRegistry. "
            "Built-in options: 'lerobot', 'custom'. "
            "Participants can register their own policies using @PolicyRegistry.register('my_policy')."
        ),
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default="outputs/train/diffusion_fold_1/checkpoints/100000/pretrained_model",
        help="Path to the pretrained IL policy checkpoint.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Path of the train dataset (for metadata).",
    )
    parser.add_argument(
        "--use_ee_pose",
        action="store_true",
        help="If set, policy outputs end-effector poses instead of joint angles. IK will be used to convert to joint angles.",
    )
    parser.add_argument(
        "--ee_urdf_path",
        type=str,
        default="Assets/robots/so101_new_calib.urdf",
        help="URDF path for IK solver (required when --use_ee_pose is set).",
    )

    return parser
