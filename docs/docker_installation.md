# Docker Installation Guide

This guide provides step-by-step instructions for installing the LeHome Challenge environment using Docker.

## Installation Steps

### 1. Install Docker

```bash
# Install Docker using the convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Post-install steps to run Docker without sudo
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

# Verify the installation
docker run hello-world
```

### 2. Download the Docker Image

```bash
wget https://huggingface.co/datasets/lehome/docker/resolve/main/lehome-challenge.tar.gz
```

> **Note:** Make sure you have sufficient disk space before downloading.

### 3. Load the Docker Image

```bash
docker load -i lehome-challenge.tar.gz
```

### 4. Run and Activate the Environment

```bash
# Start the container (adjust flags as needed)
docker run -it lehome-challenge
# Inside the container, activate the environment and verify
cd /opt/lehome-challenge
source .venv/bin/activate
```

### 5. Eval

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu \
    --headless
```

> **Note:** Make sure you enable headless mode.


## More Details

Now that you have installed the environment, you can:

- [Start Training](training.md)
- [Evaluate Policies](policy_eval.md)
- [Back to README](../README.md)
