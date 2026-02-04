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

- [📑 Table of Contents](#-table-of-contents)
- [🚀 Quick Start](#-quick-start)
  - [1. Installation](#1-installation)
    - [Use UV](#use-uv)
    - [Use Docker](#use-docker)
  - [2. Assets \& Data Preparation](#2-assets--data-preparation)
    - [Download Simulation Assets](#download-simulation-assets)
    - [Download Example Dataset](#download-example-dataset)
    - [Collect Your Own Data](#collect-your-own-data)
  - [3. Train](#3-train)
    - [Quick Start](#quick-start)
  - [4. Eval](#4-eval)
    - [Common Options](#common-options)
    - [Garment Test Configuration](#garment-test-configuration)
- [📮 Submission](#-submission)
- [🧩 Acknowledgments](#-acknowledgments)

## 🚀 Quick Start

> ⚠️ **IMPORTANT**: 
> For Ubuntu version and GPU-related settings, please refer to the [IsaacSim 5.1.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html). And the simulation currently only supports CPU devices.

### 1. Installation
We offer two installation methods: UV and Docker for submission and local evaluation.

#### Use UV

The simulation environment is based on the IssacLab and LeRobot repositories; please refer to [UV installation guide](docs/installation.md).

#### Use Docker

***Docker is not ready yet; please use uv for Installation for now.***

### 2. Assets & Data Preparation

#### Download Simulation Assets

Download the required simulation assets (scenes, objects, robots) from HuggingFace:

```bash
# This creates the Assets/ directory with all required simulation resources
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

#### Download Example Dataset

We provide demonstrations for four types of garments. Download from HuggingFace:

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

If you need depth information or individual data for each garment. Download from HuggingFace:

```bash
hf download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/example
```

#### Collect Your Own Data
For detailed instructions on teleoperation data collection and dataset processing, please refer to our [Dataset Collection and Processing Guide](docs/datasets.md) ( using SO101 Leader is strongly recommended).

### 3. Train

We provide several training examples; the models and training framework are from LeRobot.

#### Quick Start

Train using one of the pre-configured training files:

```bash
lerobot-train --config_path=configs/train_<policy>.yaml
```

**Available config files:**
- `configs/train_act.yaml` - ACT 
- `configs/train_dp.yaml` - DP
- `configs/train_smolvla.yaml` - SmolVLA 

**Key configuration options:**
- **Dataset path**: Update `dataset.root` to point to your dataset
- **Input/Output features**: Specify which observations and actions to use
- **Training parameters**: Adjust `batch_size`, `steps`, `save_freq`, etc.
- **Output directory**: Modify `output_dir` to save models elsewhere

> 📖 **For detailed training instructions, feature selection guide, and configuration options, see our [Training Guide](docs/training.md).**

### 4. Eval

Evaluate your trained policy on the challenge garments. The framework supports LeRobot policies and custom implementations.

**Examples:**

```bash
# Evaluate using LeRobot policy 
# Note: --policy_path and --dataset_root are required parameters for LeRobot policies, ready to run once the dataset and model checkpoints are prepared.
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu    

# Evaluate custom policy
# Note: Participants can define their own model loading logic within the policy class. Provides flexibility for participants to implement specialized loading and inference logic.
python -m scripts.eval \
    --policy_type custom \
    --garment_type "top_long" \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

#### Common Options

| Parameter | Description | Default | Required For |
|-----------|-------------|---------|--------------|
| `--policy_type` | Policy type: `lerobot`, `custom` | `lerobot` | All |
| `--policy_path` | Path to model checkpoint | - | All (passed as `model_path` for custom) |
| `--dataset_root` | Dataset path (for metadata) | - | **LeRobot only** |
| `--garment_type` | Type of garments: `top_long`, `top_short`, `pant_long`, `pant_short`, `custom` | `top_long` | All |
| `--num_episodes` | Episodes per garment | `5` | All |
| `--max_steps` | Max steps per episode | `600` | All |
| `--save_video` | Save evaluation videos | | All |
| `--video_dir` | Directory to save evaluation videos | `outputs/eval_videos` | `--save_video` |
| `--enable_cameras` | Enable camera rendering | | All |
| `--device` | Device for inference: only `cpu` |'cpu'| All |
| `--headless` | Used for evaluation without GUI | disabled | All |

**Parameter Descriptions:**

* **Required for LeRobot Policy**: `--policy_path` (model path) and `--dataset_root` (dataset path, used for loading metadata).
* **Custom Policy**: `--policy_path` is passed to the policy constructor as `model_path`. Participants can define their own model loading logic (refer to `scripts/eval_policy/example_participant_policy.py`).


#### Garment Test Configuration
Evaluation is performed on the `Release` set of garments. Under the directory `Assets/objects/Challenge_Garment/Release`, each garment category folder contains a corresponding text file listing the garment names (e.g., `Top_Long/Top_Long.txt`).

*   **Evaluate a Category**: Set `--garment_type` to `top_long`, `top_short`, `pant_long`, or `pant_short` to evaluate all garments within that category.
*   **Evaluate Specific Garments**: Edit `Assets/objects/Challenge_Garment/Release/Release_test_list.txt` to include only the garments you want to test, then run with `--garment_type custom`.

> 📖 **For detailed policy evaluation guide**, see [eval_guide](docs/policy_eval.md).


## 📮 Submission

Once you are satisfied with your model's performance, follow these steps to submit your results to the competition leaderboard:

>Submission instructions will be available on the [competition website](https://lehome-challenge.com/).

## 🧩 Acknowledgments

This project stands on the shoulders of giants. We utilize and build upon the following excellent open-source projects:

- **[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)** - For photorealistic physics simulation
- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html)** - For modular robot learning environments
- **[LeRobot](https://github.com/huggingface/lerobot)** - For state-of-the-art Imitation Learning algorithms
- **[Marble](https://marble.worldlabs.ai/)** - For diverse simulation scene generation
