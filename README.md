# From Microscopic Images to AI-Driven Insights: A Transparent Approach to Hematologic Cancer Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the manuscript: *"From Microscopic Images to AI-Driven Insights: A Transparent Approach to Hematologic Cancer Detection"*. This repository provides a fully reproducible, scalable PyTorch training pipeline designed for high-performance medical image analysis and leukemia classification.

## 🌟 Features

* **Distributed Training:** Out-of-the-box support for multi-GPU training using PyTorch Distributed Data Parallel (DDP).
* **State-of-the-Art Models:** Easily swap between various CNN and Vision Transformer (ViT) backbones using `timm` and `torchvision`.
* **Robust Validation:** Implements $n$-fold cross-validation with strict CNMC patient-level grouping to prevent data leakage.
* **Strict Reproducibility:** Comprehensive deterministic seeding configuration for exact experiment replication.
* **Configurable Execution:** Parameterized script using `argparse` for seamless execution and hyperparameter tuning.

## 🛠️ Installation

1. Clone the repository:
```bash
   git clone https://github.com/OmidMehdizadehD/Leukemia-Classification.git
   cd Leukemia-Classification
```

2. Install the required dependencies:
```bash
   pip install -r requirements.txt
```

## 📂 Data Preparation

The pipeline expects the dataset structured within your data directory. Due to patient-level grouping, ensure your metadata/labels can be mapped correctly to the patient IDs.
```bash
[data_dir]/
├── train/
│   ├── class_0/
│   └── class_1/
├── val/
└── test/
```

## 🚀 Usage

### Multi-GPU Training (DDP)

The training script (`train.py`) is designed to be run using `torchrun` for distributed training. You can specify the number of GPUs and hyperparameter configurations via command-line arguments.

bash
# Example: Running on 2 GPUs on a single machine (--standalone)
torchrun --standalone --nproc_per_node=2 train.py \
--data_dir /path/to/your/dataset \
--save_dir ./checkpoints \
--seed 42

### Command-Line Arguments

You can customize training runs by passing arguments (ensure these are defined in your `train.py` script's `argparse` setup):
* `--data_dir`: Path to the root directory of your dataset.
* `--save_dir`: Directory where model weights and logs will be saved.
* `--seed`: Master seed for reproducibility (default: 42).
* *(Additional arguments like `--lr`, `--batch_size`, or `--folds` can be passed if defined in the script).*

## 🔬 Reproducibility

To ensure completely reproducible results across runs, this repository utilizes a strict `set_seed()` function that fixes random seeds across Python, NumPy, PyTorch, and CUDA environment variables (`PYTHONHASHSEED`). It also configures `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. 

## 📝 Citation

This manuscript is currently under review. If you find this code useful in your research, please link to this repository:

text
Mehdizadeh, O. (2024). From Microscopic Images to AI-Driven Insights: A Transparent Approach to Hematologic Cancer Detection. GitHub Repository: https://github.com/OmidMehdizadehD/Leukemia-Classification
*(Citation format will be updated upon publication).*

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
