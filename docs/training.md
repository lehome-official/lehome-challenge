# Training Guide

This guide covers how to train policies for the LeHome Challenge, including feature selection, configuration options, and custom policy integration.

## Table of Contents

- [1. Quick Start](#1-quick-start)
- [2. Training with Official Policies](#2-training-with-official-policies)
  - [2.1 Available Policies](#21-available-policies)
  - [2.2 Basic Training Command](#22-basic-training-command)
  - [2.3 Configuration File Structure](#23-configuration-file-structure)
  - [2.4 Dataset Features](#24-dataset-features)
  - [2.5 Feature Selection](#25-feature-selection)
  - [2.6 Training Parameters](#26-training-parameters)
- [3. Bring Your Own Policy](#3-bring-your-own-policy)

---

## 1. Quick Start

Train a policy using one of the pre-configured training files:

```bash
lerobot-train --config_path=configs/train_act.yaml
```

**Available config files:**
- `configs/train_act.yaml` - ACT policy
- `configs/train_dp.yaml` - Diffusion Policy
- `configs/train_smolvla.yaml` - SmolVLA policy

---

## 2. Training with Official Policies

### 2.1 Available Policies

LeHome currently provides configuration files for the following policies:

| Policy | Type | Description | Config File |
|--------|------|-------------|-------------|
| `act` | Imitation Learning | Action Chunking Transformer | `configs/train_act.yaml` |
| `diffusion` | Imitation Learning | Diffusion Policy | `configs/train_dp.yaml` |
| `smolvla` | Vision-Language-Action | Small Vision-Language-Action Model | `configs/train_smolvla.yaml` |

> 💡 **Note**: LeRobot supports additional policies (π0, π0.5, GR00T, X-VLA), but configuration files for these are not provided in this repository. You can create custom configuration files following the [LeRobot documentation](https://huggingface.co/docs/lerobot) or use the above three baseline policies.

### 2.2 Basic Training Command

The recommended way to train is using a configuration file:

```bash
lerobot-train --config_path=path/to/your/config.yaml
```

> ⚠️ **Note:** Using configuration files (instead of command-line arguments) allows you to explicitly specify which features to use for training.

### 2.3 Configuration File Structure

A typical training configuration file looks like this:

```yaml
dataset:
  repo_id: local_dataset_129
  root: Datasets/record/129

policy:
  type: act
  device: cuda
  push_to_hub: false
  
  input_features:
    observation.state:
      type: STATE
      shape: [12]
    observation.images.top_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.left_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.right_rgb:
      type: VISUAL
      shape: [3, 480, 640]
  
  output_features:
    action:
      type: ACTION
      shape: [12]

output_dir: outputs/train/act_129
batch_size: 16
steps: 30000
save_freq: 10000
log_freq: 1000

wandb:
  enable: true
  project: lerobot_act_129
```

**Key sections:**

- **dataset**: Specifies the dataset location
- **policy**: Defines policy type, device, and input/output features
- **output_dir**: Where to save checkpoints and logs
- **Training parameters**: batch_size, steps, save_freq, log_freq
- **wandb**: Weights & Biases logging configuration

### 2.4 Dataset Features

The LeHome dataset (in maximum configuration) contains the following features:

| Feature | Shape | Description |
|---------|-------|-------------|
| `observation.state` | (12,) | Dual-arm joint positions |
| `action` | (12,) | Dual-arm joint actions |
| `observation.images.top_rgb` | (480, 640, 3) | Top camera RGB image |
| `observation.images.left_rgb` | (480, 640, 3) | Left camera RGB image |
| `observation.images.right_rgb` | (480, 640, 3) | Right camera RGB image |
| `observation.top_depth` | (480, 640) | Top camera depth map |
| `observation.ee_pose` | (16,) | Dual-arm end-effector poses (position + quaternion + gripper) |
| `action.ee_pose` | (16,) | Dual-arm end-effector action poses |
| `task` | str | Task description |

> **Note:** For single-arm tasks, `observation.state`, `action`, `observation.ee_pose`, and `action.ee_pose` have half the dimensions (6, 6, 8, 8 respectively).

> ⚠️ **Important:** Using `observation.ee_pose` and `action.ee_pose` is **not recommended** due to hardware limitations of the SO101 arm. The Inverse Kinematics (IK) solver may produce inaccurate or unstable solutions, leading to poor policy performance. **We strongly recommend using joint-space control** (`observation.state` and `action`) instead.

### 2.5 Feature Selection

You can flexibly select which features to use for training by specifying them in the `input_features` and `output_features` sections.

#### Feature Types

When configuring features, note that:
- **RGB images** (`observation.images.*_rgb`) use `type: VISUAL`
- **Depth maps** (`observation.top_depth`) use `type: STATE` (not VISUAL)
- **Joint states/poses** (`observation.state`, `observation.ee_pose`) use `type: STATE`
- **Actions** (`action`, `action.ee_pose`) use `type: ACTION`

> **Note:** `observation.top_depth` is configured as `STATE` because LeRobot's visual feature consistency validation only checks features explicitly marked as `VISUAL` (RGB images). Using `STATE` for depth maps allows more flexible configuration.

#### Verified Feature Combinations

The following input feature combinations have been verified to work:

**Combination 1: State + RGB Cameras**
```yaml
input_features:
  observation.state:
    type: STATE
    shape: [12]
  observation.images.top_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.left_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.right_rgb:
    type: VISUAL
    shape: [3, 480, 640]
```

**Combination 2: State + RGB Cameras + Depth**
```yaml
input_features:
  observation.state:
    type: STATE
    shape: [12]
  observation.images.top_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.left_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.right_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.top_depth:
    type: STATE
    shape: [1, 480, 640]
```

**Combination 3: End-Effector Pose + RGB Cameras + Depth**

> ⚠️ **Not Recommended:** This combination uses end-effector poses, which may lead to unstable performance due to IK solver limitations with the SO101 arm hardware. Use joint-space control (Combination 1 or 2) for better results.

```yaml
input_features:
  observation.ee_pose:
    type: STATE
    shape: [16]
  observation.images.top_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.left_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.images.right_rgb:
    type: VISUAL
    shape: [3, 480, 640]
  observation.top_depth:
    type: STATE
    shape: [1, 480, 640]
```

#### Using Partial Cameras

If you want to use only a subset of cameras (e.g., only `top_rgb`), you need to add a `rename_map` to bypass the visual feature consistency validation:

```yaml
policy:
  input_features:
    observation.state:
      type: STATE
      shape: [12]
    observation.images.top_rgb:   # Only using top camera
      type: VISUAL
      shape: [3, 480, 640]
  
  output_features:
    action:
      type: ACTION
      shape: [12]

# Key configuration: bypass visual feature consistency check
rename_map:
  observation.images.left_rgb: observation.images.left_rgb
  observation.images.right_rgb: observation.images.right_rgb
```

**How it works:**

- **Validation phase**: Providing `rename_map` skips the visual feature consistency check
- **Data loading phase**: All camera data is still loaded (occupies memory)
- **Model training phase**: Only features declared in `input_features` are used

**Pros:**
- No need to modify the dataset
- No need to modify LeRobot source code
- Flexible camera selection

**Cons:**
- Unused cameras still occupy memory
- But they don't participate in training computation

### 2.6 Training Parameters

#### Dataset Configuration

```yaml
dataset:
  repo_id: local_dataset_001        # Dataset identifier
  root: Datasets/record/001         # Dataset path
```

#### Policy Configuration

```yaml
policy:
  type: act                         # Policy type: act, diffusion, smolvla, etc.
  device: cuda                      # Device: cuda or cpu
  push_to_hub: false                # Whether to push to HuggingFace Hub
```

#### Training Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `batch_size` | Batch size for training | 8, 16, 32, 64 |
| `steps` | Total training steps | 20000, 30000, 50000 |
| `save_freq` | Checkpoint save frequency | 5000, 10000 |
| `log_freq` | Logging frequency | 100, 1000 |
| `learning_rate` | Learning rate (policy-specific) | 1e-4, 5e-4 |

#### Output Configuration

```yaml
output_dir: outputs/train/experiment_name   # Where to save checkpoints
```

Checkpoints will be saved to:
- `{output_dir}/checkpoints/last/pretrained_model` - Latest checkpoint
- `{output_dir}/checkpoints/step_{N}/pretrained_model` - Periodic checkpoints

#### WandB Configuration

```yaml
wandb:
  enable: true                      # Enable Weights & Biases logging
  project: my_project_name          # WandB project name
  entity: my_username               # WandB username (optional)
```

---

## 3. Bring Your Own Policy

For teams who want to integrate their own custom policies, please refer to the official LeRobot documentation:

> **Official Guide:** [Bring Your Own Policies](https://huggingface.co/docs/lerobot/bring_your_own_policies)

The guide covers:
- Policy package structure
- Configuration and processor implementation
- Model architecture integration
- Registration and usage

### Quick Reference: Package Structure

```text
lerobot_policy_my_custom_policy/
├── pyproject.toml
└── src/
    └── lerobot_policy_my_custom_policy/
        ├── __init__.py
        ├── configuration_my_custom_policy.py
        ├── modeling_my_custom_policy.py
        └── processor_my_custom_policy.py
```

Once your custom policy is properly packaged and installed, you can use it in LeHome training with:

```bash
lerobot-train --config_path=configs/train_your_policy.yaml
```

### Example: Custom Policy Configuration

```yaml
dataset:
  repo_id: local_dataset_001
  root: Datasets/record/001

policy:
  type: my_custom_policy            # Your custom policy name
  device: cuda
  
  input_features:
    # Define your input features
    observation.state:
      type: STATE
      shape: [12]
    observation.images.top_rgb:
      type: VISUAL
      shape: [3, 480, 640]
  
  output_features:
    # Define your output features
    action:
      type: ACTION
      shape: [12]
  
  # Your custom policy-specific parameters
  custom_param1: value1
  custom_param2: value2

output_dir: outputs/train/my_custom_policy
batch_size: 16
steps: 30000
```

For specific questions about custom policy integration with the LeHome environment, please open an issue on our repository.

---

## Additional Resources

- [Dataset Collection and Processing Guide](datasets.md)
- [Installation Guide](installation.md)
- [LeRobot Official Documentation](https://huggingface.co/docs/lerobot)
- [Competition Website](https://lehome-challenge.com/)

