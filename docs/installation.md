# Manual Installation Guide

This guide provides step-by-step instructions for manually installing the LeHome Challenge environment.

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- The GPU driver and CUDA follow the official IsaacLab tutorial.

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

### 2. Install Dependencies with uv

```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

### 3. Clone and Configure IsaacLab

```bash
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
```

### 4. Install IsaacLab

Activate the virtual environment and install IsaacLab:

```bash
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

### 5. Install LeHome Package

Finally, install the LeHome package in development mode:

```bash
uv pip install -e ./source/lehome
```

---

## Next Steps

Now that you have installed the environment, you can:

- [Prepare Assets and Data](datasets.md)
- [Start Training](training.md)
- [Evaluate Policies](../scripts/eval_policy/POLICY_GUIDE.md)
- [Back to README](../README.md)
