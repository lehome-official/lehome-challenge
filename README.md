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
[![LeRobot](https://img.shields.io/badge/LeRobot-0.4.1-yellow.svg)](https://github.com/huggingface/lerobot)
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

### 1. Installation
We recommend using our official Docker image for development and evaluation to ensure reproducible and consistent environments.

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

#### Advanced: Manual Installation (Step-by-step)

If you prefer to set up the environment manually or want more customization, please refer to our [step-by-step installation guide](docs/installation.md).

### 2. Assets & Data Preparation

#### Download Simulation Assets

Download the required simulation assets (scenes, objects, robots) from HuggingFace:

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

#### Download Example Dataset

We provide example demonstration data collected on Release assets. Download from HuggingFace:

```bash
hf download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/record/example
```

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

```bash
# Evaluate LeRobot policy on release stage
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_fold/checkpoints/100000/pretrained_model \
    --dataset_root Datasets/record/001 \
    --stage release \
    --num_episodes 5

# Evaluate custom policy on single garment
python -m scripts.eval \
    --policy_type custom \
    --policy_path path/to/model.pth \
    --stage single \
    --garment_name Top_Long_Unseen_0 \
    --num_episodes 5
```

#### Evaluation Stages

- `--stage release`: Evaluate on all release garments (default)
- `--stage holdout`: Evaluate on holdout garments (for final testing)
- `--stage single`: Evaluate on a specific garment (specify with `--garment_name`)

#### Common Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--policy_type` | Policy type: `lerobot`, `custom`, `scripted` | `lerobot` |
| `--policy_path` | Path to model checkpoint | Required |
| `--dataset_root` | Dataset path (for metadata, LeRobot only) | Required for LeRobot |
| `--num_episodes` | Episodes per garment | `5` |
| `--max_steps` | Max steps per episode | `600` |
| `--save_video` | Save evaluation videos | `False` |

> 📖 **For detailed policy integration guide**, see [scripts/eval_policy/POLICY_GUIDE.md](scripts/eval_policy/POLICY_GUIDE.md)


## 📮 Submission

Once you are satisfied with your model's performance, follow these steps to submit your results to the competition leaderboard:

>Submission instructions will be available on the [competition website](https://lehome-challenge.com/).

## 🧩 Acknowledgments

This project stands on the shoulders of giants. We utilize and build upon the following excellent open-source projects:

- **[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)** - For photorealistic physics simulation
- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html)** - For modular robot learning environments
- **[LeRobot](https://github.com/huggingface/lerobot)** - For state-of-the-art Imitation Learning algorithms

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