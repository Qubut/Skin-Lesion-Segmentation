# Skin Lesion Segmentation with UNet

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

## Overview
Medical image segmentation pipeline for skin lesion detection using:
- 🧠 Modified UNet architecture with attention gates
- ⚡ PyTorch Lightning for training orchestration
- 🧪 Optuna for hyperparameter optimization
- 📊 Weights & Biases/TensorBoard for experiment tracking

## Key Features
- Multi-scale context aggregation via ASPP module
- Residual connections for stable training
- Attention gates for focused feature learning
- Elastic transformations for data augmentation
- Mixed-precision training support
- Reproducible experiments via Hydra config


## Installation

### 1. System Requirements
- Ubuntu 22.04 LTS or Windows 11 with WSL2 (Ubuntu 22.04)
- NVIDIA GPU with CUDA 12.8 support (optional)

### 2. Nix & direnv Setup

#### Ubuntu
```bash
# Install Nix package manager
sh <(curl -L https://nixos.org/nix/install) --daemon

# Install direnv
nix-env -iA nixpkgs.direnv

# Setup direnv integration
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

#### WSL2 (Windows)
```powershell
# Install Nix (from PowerShell)
iex (irm https://get.scoop.sh)
scoop install nix

# Install direnv
scoop install direnv

# Setup WSL integration
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Project Setup
```bash
# Clone repository
git clone https://github.com/yourname/skin-lesion-segmentation
cd skin-lesion-segmentation

# Allow direnv (auto-activates Nix environment)
direnv allow

# Verify installation
nix --version  # Should show 2.18.0+
direnv status # Should show "Loaded"
```


## Usage

### Dataset Preparation

Organize your data directory as:

```
data/
├── train/
│   ├── image1.jpg
│   └── image2.jpg
├── train_mask/
│   ├── image1*.png```markdown
### Training Modes

#### 1. Standard Training

```bash
python train.py \
  data=img_size=256 \
  training.batch_size=32 \
  training.augmentation=True
```

#### 2. Hyperparameter Optimization

```bash
python train.py \
  optimize=True \
  optuna.n_trials=50 \
  optuna.storage="sqlite:///optuna.db"
```

#### 3. Use Optimized Parameters

```bash
python train.py \
  data.img_size=256 \
  training.batch_size=64 \
  training.augmentation=True \
  model.learning_rate=0.000316
```

## Configuration

Modify `config.yml` to customize:

```yaml
data:
  data_path: "./data"
  train_split: 0.8

model:
  num_classes: 1
  learning_rate: 0.001

training:
  max_epochs: 100
  accelerator: "gpu"
```

## Directory Structure

```
├── configs/         # Hydra configuration files
├── data/            # Dataset directory
├── logs/            # Training logs
├── models/          # Model implementations
├── notebooks/       # Experiment notebooks
└── outputs/         # experiments outputs including checkpoints (w/ & w/o optimization)
```

## Visualization

Training metrics are logged to:

- outputs/date/timestamp/logs
- outputs/date/timestamp/train.log

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## References

1. Ronneberger et al. [UNet Paper](https://arxiv.org/abs/1505.04597)
2. PyTorch Lightning [Documentation](https://lightning.ai/docs/pytorch/stable/)
3. Optuna [Hyperparameter Optimization](https://optuna.readthedocs.io/)
