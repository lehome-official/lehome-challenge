# Dataset Collection and Processing Guide

This guide covers how to collect teleoperation demonstration data and process datasets for training.

## Table of Contents

- [Dataset Collection and Processing Guide](#dataset-collection-and-processing-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Hardware Setup](#2-hardware-setup)
    - [2.1 Hardware Connection](#21-hardware-connection)
    - [2.2 Calibration](#22-calibration)
    - [2.3 Verify Calibration](#23-verify-calibration)
  - [3. Teleoperation Data Collection](#3-teleoperation-data-collection)
    - [3.1 Basic Recording Command](#31-basic-recording-command)
    - [3.2 Parameter Configuration](#32-parameter-configuration)
    - [3.3 Recording Controls](#33-recording-controls)
  - [4. Dataset Inspection and Processing](#4-dataset-inspection-and-processing)
    - [4.1 Inspect Dataset](#41-inspect-dataset)
    - [4.2 Read Dataset States](#42-read-dataset-states)
    - [4.3 Add End-Effector Pose](#43-add-end-effector-pose)
    - [4.4 Add PointCloud](#44-add-pointcloud)
    - [4.5 Replay Dataset](#45-replay-dataset)
    - [4.6 Remove Features](#46-remove-features)
  - [5. Dataset Merging](#5-dataset-merging)
  - [6. Dataset Format](#6-dataset-format)
    - [6.1 Data Features](#61-data-features)
    - [6.2 Metadata](#62-metadata)
  - [7. Best Practices](#7-best-practices)

## 1. Overview

The LeHome Challenge supports collecting demonstration data through teleoperation using keyboard or SO101 Leader Arms. The collected data is saved in LeRobot dataset format and can be processed and merged for training.

## 2. Hardware Setup

This section covers the setup and configuration of SO101 Leader Arms for teleoperation.

### 2.1 Hardware Connection

For **Dual SO101 Leader Arms** teleoperation:

1. **Physical Connection**
   - Connect both Leader Arms to the computer via USB
   - Ensure both arms are powered on

2. **Identify Serial Devices**
   ```bash
   ls /dev/ttyACM*
   # Usually displays: /dev/ttyACM0, /dev/ttyACM1
   ```

3. **Grant Serial Permissions**
   ```bash
   sudo chmod 666 /dev/ttyACM0
   sudo chmod 666 /dev/ttyACM1
   ```

4. **Determine Left/Right Mapping**
   - Note which serial port corresponds to left/right arm
   - **Note:** Usually, the USB port plugged in first is assigned `/dev/ttyACM0`
   - Example: Left Arm: `/dev/ttyACM0`, Right Arm: `/dev/ttyACM1`

### 2.2 Calibration

**First-time use** or after **changing hardware**, calibrate the SO101 Leader Arms:

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --device "cpu" \
    --recalibrate \
    --enable_cameras
```

**Calibration Steps:**

1. Run the command above; the program will prompt for calibration
2. **Left Arm Calibration**: Move the left arm to maximum/minimum positions for each joint to record joint limits
3. **Right Arm Calibration**: Repeat the process for the right arm
4. Calibration data is saved automatically
5. Press **Ctrl+C** to exit

**Calibration File Location:**

Calibration data is saved in:
```
source/lehome/lehome/devices/lerobot/.cache/
├── left_arm_calibration.json
└── right_arm_calibration.json
```

**Note:** Calibration is required only once. Recalibrate if hardware is replaced or control feels inaccurate.

### 2.3 Verify Calibration

Test the hardware calibration without recording:

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --device "cpu" \
    --enable_cameras
```

**Verification Steps:**

1. IsaacSim window opens displaying the scene
2. Move the Leader Arm and verify the simulated robot follows correctly
3. Confirm left/right arm mapping is correct
4. Press **Ctrl+C** to exit

## 3. Teleoperation Data Collection

### 3.1 Basic Recording Command

**Recommended command for dual-arm data collection:**

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --garment_name Top_Long_Unseen_0 \
    --enable_record \
    --num_episode 10 \
    --log_success \
    --device "cpu" \
    --enable_cameras
```

### 3.2 Parameter Configuration

**Input Device Options:**

| Parameter | Single-Arm | Dual-Arm | Description |
|-----------|-----------|----------|-------------|
| `--teleop_device` | `keyboard` | `bi-keyboard` | Keyboard control |
| `--teleop_device` | `so101leader` | `bi-so101leader` | SO101 Leader device |
| `--port` | `/dev/ttyACM0` | - | Single-arm serial port |
| `--left_arm_port` | - | `/dev/ttyACM0` | Dual-arm left port |
| `--right_arm_port` | - | `/dev/ttyACM1` | Dual-arm right port |
| `--recalibrate` | Optional | Optional | Calibrate on first use |

**Task Configuration:**

| Parameter | Example | Description |
|-----------|---------|-------------|
| `--task` | `LeHome-SO101-Direct-Garment-v2` | Single-arm task |
| `--task` | `LeHome-BiSO101-Direct-Garment-v2` | Dual-arm task |
| `--garment_name` | `Top_Long_Unseen_0` | Garment identifier |
| `--garment_version` | `Release` / `Holdout` | Asset version |
| `--task_description` | `"fold the garment"` | Task label for filtering |

**Recording Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_record` | Required | Enable dataset recording |
| `--num_episode` | 20 | Number of episodes |
| `--step_hz` | 120 | Environment stepping rate (Hz) |
| `--log_success` | False | Enable real-time success checking and distance logging (useful for monitoring task progress) |
| `--disable_depth` | False | Disable depth maps (faster) |
| `--enable_pointcloud` | False | Please convert depth to pointcloud offline following [Add pointcloud](#44-add-pointcloud) |
| `--record_ee_pose` | False | Record end-effector pose |
| `--ee_urdf_path` | None | URDF path (required for `--record_ee_pose`) |
| `--ee_state_unit` | `rad` | Joint angle unit (`rad` or `deg`) |

**Simulation Options (AppLauncher):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_cameras` | False | Enable camera rendering |
| `--headless` | False | Run in headless mode (no GUI) |
| `--device` | `cpu` |  Use `cpu` to avoid garment physics issues |

### 3.3 Recording Controls

**Function Keys:**
- **B key**: Start teleoperation (activate control, must be pressed before `S`)
- **S key**: Start recording current episode
- **N key**: Save current episode (mark as success)
- **D key**: Discard current episode (re-record)
- **ESC key**: Abort recording and clear buffer
- **Ctrl+C**: Exit program

**Keyboard Control (for keyboard/bi-keyboard):**
- **Single-arm**: WASD (position), QE (Z-axis), Arrow keys (orientation), Space/Shift (gripper)
- **Dual-arm**: Letter keys (T/G, Y/H, U/J, I/K, O/L, Q/A) for left arm, Number keys (1/2, 3/4, 5/6, 7/8, 9/0, [/] or -/+) for right arm

**SO101 Leader Control:**
- Directly move the physical Leader Arms; the simulated robots follow in real-time

## 4. Dataset Inspection and Processing

### 4.1 Inspect Dataset

View dataset metadata, frame data, and statistics:

```bash
python -m scripts.dataset inspect \
    --dataset_root Datasets/record/001
```

**Additional Parameters:**

- `--show_frames N`: Display first N frames of sample data
- `--show_stats`: Display detailed statistical information for numeric columns

### 4.2 Read Dataset States

Read and analyze dataset observation/action data:

```bash
python -m scripts.dataset read \
    --dataset_root Datasets/record/001 \
    --num_frames 10
```

**Additional Parameters:**

- `--num_frames N`: Number of frames to display
- `--episode N`: Read specific episode index
- `--output_csv PATH`: Export data to CSV file
- `--show_stats`: Display statistical information

### 4.3 Add End-Effector Pose

Add end-effector pose to existing datasets offline (computes both `observation.ee_pose` and `action.ee_pose`):

```bash
python -m scripts.dataset augment \
    --dataset_root Datasets/record/001 \
    --urdf_path Assets/robots/so101_new_calib.urdf \
    --state_unit rad \
    --overwrite
```

**Parameters:**
- `--urdf_path`: Robot URDF file path
- `--state_unit`: Joint angle unit (`rad` or `deg`)
- `--overwrite`: Overwrite existing EE pose data if present

**Output:**
- Single-arm: 8 dimensions (position xyz + quaternion xyzw + gripper)
- Dual-arm: 16 dimensions (left arm 8D + right arm 8D)

### 4.4 Add PointCloud
Add pointcloud to existing datasets offline (result in  `XYZRGB`):

```bash
python scripts/utils/process_parquet_to_pc.py \
    --dataset_root Datasets/record/001 \
    --num_points 4096
```

**Parameters:**
- `--dataset_root`: Dataset root path
- `--num_points`: number of points of the pointcloud

**Output:**
- Create `pointclouds` folder which contains all the episode folders, `episode_000` folder stores pointcloud for each frame, like `frame_000000.npz`

### 4.5 Replay Dataset

Replay recorded datasets for visualization, verification, or data augmentation. Supports joint angle actions or end-effector pose control (via IK).

**Example Command:**

```bash
python -m scripts.dataset_sim replay \
    --dataset_root Datasets/record/example/record_top_long_release_10/001 \
    --num_replays 1 \
    --disable_depth \
    --device "cpu" \
    --enable_cameras
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--task` | str | `LeHome-BiSO101-Direct-Garment-v2` | Task environment name |
| `--dataset_root` | str | `Datasets/record/001` | Input dataset directory |
| `--output_root` | str | `None` | Output directory (Should be specified (e.g, `Datasets/replay_stored/example/record_top_long_release_10`) if `--save_successful_only` is true) |
| `--num_replays` | int | `1` | Number of replays per episode |
| `--save_successful_only` | flag | `False` | Only save episodes that achieve success |
| `--start_episode` | int | `0` | Starting episode index (inclusive) |
| `--end_episode` | int | `None` | Ending episode index (exclusive, None = all episodes) |
| `--step_hz` | int | `60` | Environment stepping rate in Hz |
| `--use_random_seed` | flag | `False` | Use random seed (no fixed seed) |
| `--seed` | int | `42` | Random seed for environment (ignored if `--use_random_seed` is set) |
| `--task_description` | str | `fold the garment on the table` | Task description string |
| `--garment_cfg_base_path` | str | `Assets/objects/Challenge_Garment` | Base path of the garment configuration |
| `--particle_cfg_path` | str | `source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml` | Path of the particle configuration |
| `--disable_depth` | flag | `False` | Disable depth observation (faster replay) |
| `--use_ee_pose` | flag | `False` | Use `action.ee_pose` control (Cartesian space, converted via IK) |
| `--ee_urdf_path` | str | `Assets/robots/so101_new_calib.urdf` | URDF file path (required for `--use_ee_pose`) |
| `--ee_state_unit` | str | `rad` | Joint angle unit: `rad` or `deg` |
| `--device` | str | `cuda:0` | Simulation device (`cpu` or `cuda:0`). Use `cpu` to avoid garment physics issues |

**Notes:**

- **Pure Replay Mode**: Omit `--output_root` to replay without saving (visualization only)
- **Data Augmentation**: Use `--num_replays N` with `--save_successful_only` to extract successful trajectories
- **End-Effector Pose Control**: Requires dataset with `action.ee_pose` field (recorded with `--record_ee_pose` or augmented using `augment_ee_pose`)
- **IK Statistics**: When using `--use_ee_pose`, IK success rate and error statistics are displayed

### 4.6 Remove Features

Remove specific features from datasets (e.g., depth maps to reduce storage):

**Remove a single feature:**

```bash
lerobot-edit-dataset \
    --repo_id record_top_long_release_10/001 \
    --root Datasets/record/record_top_long_release_10/001 \
    --new_repo_id record_top_long_release_10/001_no_depth \
    --operation.type remove_feature \
    --operation.feature_names "['observation.top_depth']"
```

**Batch remove features from multiple datasets:**

```bash
cd Datasets/record/record_top_long_release_10

for i in {001..010}; do
    lerobot-edit-dataset \
        --repo_id record_top_long_release_10/$i \
        --root Datasets/record/record_top_long_release_10/$i \
        --new_repo_id record_top_long_release_10/${i}_no_depth \
        --operation.type remove_feature \
        --operation.feature_names "['observation.top_depth']"
done
```

**Parameters:**
- `--repo_id`: Dataset identifier. Used as a label in metadata and for downloading from HuggingFace Hub if local files are not found. When `--root` is specified, it doesn't affect the actual file path.
- `--root`: Dataset root directory path (relative or absolute). This determines where the dataset files are actually located.
- `--new_repo_id`: New dataset identifier (optional, if omitted, original dataset is renamed to `_old`)
- `--operation.type`: Operation type, use `remove_feature`
- `--operation.feature_names`: List of feature names to remove (Python list format)

**Notes:**
- Cannot remove required features: `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`
- If `--new_repo_id` is specified, original dataset remains unchanged
- Multiple features can be removed: `"['feature1', 'feature2']"`

## 5. Dataset Merging

Merge multiple datasets collected from different sessions into a single unified dataset.

**Using Python script:**

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.utils import init_logging

init_logging()

# Specify paths to datasets you want to merge
dataset_paths = [
    Path("Datasets/record/record_top_long_release_10_no_depth_all/001_no_depth"),
    Path("Datasets/record/record_top_long_release_10_no_depth_all/002_no_depth"),
    Path("Datasets/record/record_top_long_release_10_no_depth_all/003_no_depth"),
    # Add more dataset paths as needed
]

# Output configuration
output_dir = Path("Datasets/record/top_long_merged")
output_repo_id = "top_long_merged"

# Load datasets
datasets = [
    LeRobotDataset(f"ds_{i:03d}", root=p) 
    for i, p in enumerate(dataset_paths, 1)
]

# Merge datasets
merged = merge_datasets(datasets, output_repo_id, output_dir)
print(f"✓ Merged! Episodes: {merged.meta.total_episodes}, Frames: {merged.meta.total_frames}")
```

**One-liner command:**

```bash
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.utils import init_logging

init_logging()

# Specify dataset paths to merge
dataset_paths = [
    Path('Datasets/record/record_top_long_release_10_no_depth_all/001_no_depth'),
    Path('Datasets/record/record_top_long_release_10_no_depth_all/002_no_depth'),
    Path('Datasets/record/record_top_long_release_10_no_depth_all/003_no_depth'),
]

output_dir = Path('Datasets/record/top_long_merged')
datasets = [LeRobotDataset(f'ds_{i:03d}', root=p) for i, p in enumerate(dataset_paths, 1)]

merged = merge_datasets(datasets, 'top_long_merged', output_dir)
print(f'✓ Merged! Episodes: {merged.meta.total_episodes}, Frames: {merged.meta.total_frames}')
"
```

**Notes:**

- All source datasets must have **identical feature structures** (except for removed features)
- Merged dataset contains all episodes from source datasets in sequential order
- Video files are automatically split into multiple files based on size limit (default: 200MB)
- Data files are automatically split into multiple files based on size limit (default: 100MB)

## 6. Dataset Format

Recorded data is saved in LeRobot dataset format with the following structure:

```
Datasets/record/001/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet         
│       └── file-001.parquet          
├── images/
│   ├── observation.images.top_rgb   
│   ├── observation.images.left_rgb
│   └── observation.images.right_rgb
├── meta/
│   ├── episodes/
│   │   └── chunk-000/
│   │       └── file-000.parquet      
│   ├── garment_info.json             
│   ├── info.json                     
│   ├── stats.json                    
│   └── tasks.parquet                  
├── videos/
│   ├── observation.images.top_rgb    
│   ├── observation.images.left_rgb
│   └── observation.images.right_rgb
└── pointclouds/
    ├── episode_000
    │   ├── frame_000000.npz
    │   ├── frame_000001.npz
    │   └── frame_000002.npz
    └── episode_001
```

### 6.1 Data Features

- **observation.state**: Joint positions (6D for single-arm, 12D for dual-arm)
- **action**: Joint actions (same dimension as state)
- **observation.images.{camera}**: RGB images from different camera views
- **observation.top_depth**: Depth map from top camera (if enabled)
- **observation.ee_pose**: End-effector pose (if `--record_ee_pose` enabled)
- **task**: Task description string

### 6.2 Metadata

The `meta/` directory contains several metadata files:

- **`info.json`**: Dataset metadata including total episodes, total frames, feature definitions, and dataset configuration
- **`garment_info.json`**: Custom metadata file containing initial pose information for each episode:

  ```json
  {
    "Top_Long_Unseen_0": {
      "0": {
        "object_initial_pose": [...],
        "scale": [...],
        "garment_name": "Top_Long_Unseen_0"
      },
      "1": {...}
    }
  }
  ```

- **`stats.json`**: Statistical information about the dataset (e.g., mean, std for numeric features)
- **`tasks.parquet`**: Task descriptions for each episode in Parquet format
- **`episodes/`**: Episode-level metadata stored in Parquet format, organized by chunks

## 7. Best Practices

1. **Consistent Task Description**: Use the same `--task_description` for similar tasks to enable better filtering during training
2. **Episode Quality**: Use the **D key** to discard low-quality episodes during recording
3. **Recording Frequency**: Use `--step_hz 120` for smooth recordings. Lower values may miss fast motions
4. **Dataset Organization**: Record different garment types or tasks in separate dataset directories, then merge them later
5. **Backup**: Always backup your datasets before merging or processing