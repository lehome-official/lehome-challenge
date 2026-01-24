<p align="center">
  <h1 align="center">
    LeHome Challenge 2026
  </h1>
  <h2 align="center">
    Challenge on Garment Manipulation Skill Learning in Household Scenarios
  </h2>

  <h3 align="center">
    <a href="https://lehome-challenge.com/">Competition Website</a>
  </h3>
</p>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.1-green.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![LeRobot](https://img.shields.io/badge/LeRobot-0.4.2-yellow.svg)](https://github.com/huggingface/lerobot)
[![License](https://img.shields.io/badge/license-Apache%202.0-red.svg)](LICENSE)
[![ICRA](https://img.shields.io/badge/ICRA-2026-orange.svg)](https://2026.ieee-icra.org/program/competitions/)

</div>

## 📑 Table of Contents

- [Quick Start](#-quick-start)
  - [Installation](#1-installation)
  - [Assets & Data Preparation](#2-assets--data-preparation)
  - [Train](#3-train)
  - [Eval](#4-eval)
- [Submission](#-submission)
- [Acknowledgments](#-acknowledgments)
- [Citation](#-citation)

## 🚀 Quick Start

> ⚠️ **IMPORTANT**: Before starting, you must download the simulation assets and example datasets from HuggingFace. See [Step 2: Assets & Data Preparation](#2-assets--data-preparation) for instructions.

### 1. Installation
We recommend using our official Docker image for development and evaluation to ensure reproducible and consistent environments.
***But!!! Docker is not ready yet; please use uv for Manual Installation for now.***

#### Recommended: Use Docker (Quick Start)

1. **Pull the official competition Docker image**
   ```bash
   docker pull lehome/competition:latest
   ```

2. **Run the container (with all GPU support)**
   ```bash
   docker run --gpus all -it --network=host \
     -v $(pwd):/workspace/lehome \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -e DISPLAY=$DISPLAY \
     lehome/competition:latest
   ```

#### Manual Installation (Step-by-step)

If you prefer to set up the environment manually or want more customization, please refer to our [step-by-step installation guide](docs/installation.md).

### 2. Assets & Data Preparation

#### Download Simulation Assets

Download the required simulation assets (scenes, objects, robots) from HuggingFace:

```bash
# This creates the Assets/ directory with all required simulation resources
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

#### Download Example Dataset

We provide example demonstration data collected on Release assets. Download from HuggingFace:

```bash
hf download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/record/example
```

This step is optional if you plan to collect your own data.

#### Collect Your Own Data

For detailed instructions on teleoperation data collection and dataset processing, please refer to our [Dataset Collection and Processing Guide](docs/datasets.md).


### 3. Train

We provide three baseline algorithms: **ACT**, **Diffusion Policy**, and **SmolVLA**. 

> 📖 **For detailed training instructions, feature selection guide, and configuration options, see our [Training Guide](docs/training.md).**

#### Quick Start

Train using one of the pre-configured training files:

```bash
lerobot-train --config_path=configs/train_<policy>.yaml
```

**Available config files:**
- `configs/train_act.yaml` - ACT policy
- `configs/train_dp.yaml` - Diffusion Policy  
- `configs/train_smolvla.yaml` - SmolVLA policy

**Key configuration options:**
- **Dataset path**: Update `dataset.root` to point to your dataset
- **Input/Output features**: Specify which observations and actions to use
- **Training parameters**: Adjust `batch_size`, `steps`, `save_freq`, etc.
- **Output directory**: Modify `output_dir` to save models elsewhere
- **WandB settings**: Configure `wandb.enable` and `wandb.project`

See [Training Guide](docs/training.md) for complete configuration examples and feature selection strategies.

### 4. Eval

Evaluate your trained policy on the challenge garments. The framework supports LeRobot policies (ACT, Diffusion, VLA) and custom implementations.

#### Quick Start

**Examples:**

```bash
# Evaluate LeRobot policy (Recommended)
# Note: --policy_path and --dataset_root are required parameters for LeRobot policies, ready to run once the dataset and model checkpoints are prepared.
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_fold/checkpoints/100000/pretrained_model \
    --stage release \
    --garment_type "tops_long" \
    --dataset_root Datasets/record/example/record_top_long_release_10/001 \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu

# Evaluate custom policy
# Note: Participants can define their own model loading logic within the policy class. Provides flexibility for participants to implement specialized loading and inference logic.
python -m scripts.eval \
    --policy_type custom \
    --stage release \
    --garment_type "tops_long" \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

#### Common Options

| Parameter | Description | Default | Required For |
|-----------|-------------|---------|--------------|
| `--policy_type` | Policy type: `lerobot`, `custom` | `lerobot` | All |
| `--policy_path` | Path to model checkpoint | - | **LeRobot only** |
| `--dataset_root` | Dataset path (for metadata) | - | **LeRobot only** |
| `--stage` | Evaluation stage: `release`, `holdout`, `all` | `release` | All |
| `--garment_type` | Type of garments: `tops_long`, `tops_short`, `trousers_long`, `trousers_short`, `custom` | `tops_long` | All |
| `--num_episodes` | Episodes per garment | `5` | All |
| `--max_steps` | Max steps per episode | `600` | All |
| `--save_video` | Save evaluation videos | `False` | All |
| `--video_dir` | Directory to save evaluation videos | `outputs/eval_videos` | `--save_video` |
| `--enable_cameras` | Enable camera rendering | `False` | All |
| `--device` | Device for inference: `cpu`, `cuda` | `cuda` | All |
| `--use_ee_pose` | Use end-effector pose control | `False` | All |
| `--ee_urdf_path` | Robot URDF for IK solver | `Assets/robots/so101_new_calib.urdf` | `--use_ee_pose` |

**Parameter Descriptions:**

* **Required for LeRobot Policy**: `--policy_path` (model path) and `--dataset_root` (dataset path, used for loading metadata).
* **Custom Policy**: `--policy_path` is not required; participants are free to define their own model loading logic within their policy class (refer to `scripts/eval_policy/example_participant_policy.py`).


#### Garment Test Configuration
Under the directory `Assets/objects/Challenge_Garment/Release`, each garment category folder contains a corresponding text file listing the garment names (e.g., `Tops_Long/Tops_Long.txt` contains Top_Long_Seen_0, Top_Long_Seen_1, etc.). You can set --garment_type to `tops_long` to evaluate all garments within the `Tops_Long` category. The same logic applies to all other garment types.

To test a specific subset of garments, you can modify `Assets/objects/Challenge_Garment/Release/Release_test_list.txt` by including only the target garment names according to `Tops_Long.txt`, `Tops_Short.txt`, `Trousers_Long.txt`, `Trousers_Short.txt`.

> 📖 **For detailed policy integration guide**, see [scripts/eval_policy/POLICY_GUIDE.md](scripts/eval_policy/POLICY_GUIDE.md)


## 📮 Submission

Once you are satisfied with your model's performance, follow these steps to submit your results to the competition leaderboard:

>Submission instructions will be available on the [competition website](https://lehome-challenge.com/).

## 🧩 Acknowledgments

This project stands on the shoulders of giants. We utilize and build upon the following excellent open-source projects:

- **[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)** - For photorealistic physics simulation
- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html)** - For modular robot learning environments
- **[LeRobot](https://github.com/huggingface/lerobot)** - For state-of-the-art Imitation Learning algorithms
- **[Marble](https://marble.worldlabs.ai/)** - For diverse simulation scene generation

## 🖊️ Citation

If you use this framework for your research or participate in the competition, please cite our work:

```bibtex
@inproceedings{
li2025lehome,
title={LeHome: A Simulation Environment for Deformable Object Manipulation in Household Scenarios},
author={Zeyi Li and Jade Yang and Jingkai Xu and Shangbin Xie and Yuran Wang and Zhenhao Shen and Tianxing Chen and Yan Shen and Wenjun Li and Yukun Zheng and Chaorui Zhang and Ming Chen and Chen Xie and Ruihai Wu},
booktitle={IROS 2025 - 5th Workshop on RObotic MAnipulation of Deformable Objects: holistic approaches and challenges forward},
year={2025},
url={https://openreview.net/forum?id=rEDd1HorJl}
}
```
