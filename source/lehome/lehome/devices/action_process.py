import torch
from typing import Any
import numpy as np
import isaaclab.envs.mdp as mdp

from lehome.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS


def init_action_cfg(action_cfg, device):
    if device in ["so101leader"]:
        action_cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ["keyboard"]:
        action_cfg.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=0.7,
        )
    elif device in ["bi-so101leader"]:
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ["bi-keyboard", "bi-gamepad", "bi-vision"]:
        action_cfg.left_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="left_arm",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["gripper"],
            scale=0.7,
        )
        action_cfg.right_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="right_arm",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["gripper"],
            scale=0.7,
        )

    else:
        action_cfg.arm_action = None
        action_cfg.gripper_action = None
    return action_cfg


joint_names_to_motor_ids = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}


def convert_action_from_so101_leader(
    joint_state: dict[str, float],
    motor_limits: dict[str, tuple[float, float]],
    teleop_device,
) -> torch.Tensor:
    processed_action = torch.zeros(
        teleop_device.env.num_envs, 6, device=teleop_device.env.device
    )
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    for joint_name, motor_id in joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (
            motor_limit_range[1] - motor_limit_range[0]
        ) * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
        processed_radius = (
            processed_degree / 180.0 * torch.pi
        )  # convert degree to radius
        processed_action[:, motor_id] = processed_radius
    return processed_action


def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    if action.get("so101_leader") is not None:
        processed_action = convert_action_from_so101_leader(
            action["joint_state"], action["motor_limits"], teleop_device
        )
    elif action.get("keyboard") is not None:
        current_joint_pos = teleop_device.env.robot.data.joint_pos[:, :6]  # First 6 joints

        # get relative delta
        relative_delta = action["joint_state"]
        if isinstance(relative_delta, np.ndarray):
            relative_delta = torch.tensor(
                relative_delta, device=teleop_device.env.device, dtype=torch.float32
            )
        elif not isinstance(relative_delta, torch.Tensor):
            relative_delta = torch.tensor(
                relative_delta, device=teleop_device.env.device, dtype=torch.float32
            )

        # ensure dimension matches
        if relative_delta.dim() == 1:
            relative_delta = relative_delta.unsqueeze(0).expand_as(current_joint_pos)

        # add relative delta to current joint position, get absolute position
        processed_action = current_joint_pos + relative_delta

    elif action.get("bi_so101_leader") is not None:
        processed_action = torch.zeros(
            teleop_device.env.num_envs, 12, device=teleop_device.env.device
        )
        processed_action[:, :6] = convert_action_from_so101_leader(
            action["joint_state"]["left_arm"],
            action["motor_limits"]["left_arm"],
            teleop_device,
        )
        processed_action[:, 6:] = convert_action_from_so101_leader(
            action["joint_state"]["right_arm"],
            action["motor_limits"]["right_arm"],
            teleop_device,
        )
    elif action.get("bi_keyboard") is not None:
        # dual arm keyboard control: convert relative delta to absolute position
        # get current joint position (radians)
        left_current_joint_pos = teleop_device.env.left_arm.data.joint_pos[
            :, :6
        ]  # left arm first 6 joints
        right_current_joint_pos = teleop_device.env.right_arm.data.joint_pos[
            :, :6
        ]  # right arm first 6 joints

        # get relative delta
        left_relative_delta = action["joint_state"]["left_arm"]
        right_relative_delta = action["joint_state"]["right_arm"]

        # convert to torch tensor
        if isinstance(left_relative_delta, np.ndarray):
            left_relative_delta = torch.tensor(
                left_relative_delta,
                device=teleop_device.env.device,
                dtype=torch.float32,
            )
        elif not isinstance(left_relative_delta, torch.Tensor):
            left_relative_delta = torch.tensor(
                left_relative_delta,
                device=teleop_device.env.device,
                dtype=torch.float32,
            )

        if isinstance(right_relative_delta, np.ndarray):
            right_relative_delta = torch.tensor(
                right_relative_delta,
                device=teleop_device.env.device,
                dtype=torch.float32,
            )
        elif not isinstance(right_relative_delta, torch.Tensor):
            right_relative_delta = torch.tensor(
                right_relative_delta,
                device=teleop_device.env.device,
                dtype=torch.float32,
            )

        # ensure dimension matches
        if left_relative_delta.dim() == 1:
            left_relative_delta = left_relative_delta.unsqueeze(0).expand_as(
                left_current_joint_pos
            )
        if right_relative_delta.dim() == 1:
            right_relative_delta = right_relative_delta.unsqueeze(0).expand_as(
                right_current_joint_pos
            )

        # add relative delta to current joint position, get absolute position
        processed_action = torch.zeros(
            teleop_device.env.num_envs, 12, device=teleop_device.env.device
        )
        processed_action[:, :6] = left_current_joint_pos + left_relative_delta
        processed_action[:, 6:] = right_current_joint_pos + right_relative_delta

    else:
        raise NotImplementedError(
            "Only teleoperation with so101_leader, bi_so101_leader, bi_keyboard, keyboard is supported for now."
        )
    return processed_action
