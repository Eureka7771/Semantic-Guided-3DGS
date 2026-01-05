# Semantic-Guided 3D Gaussian Splatting for Resource-Efficient Indoor Scene Reconstruction

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Custom-green.svg)](LICENSE.md)

**Author:** Huang Yucheng  
**Supervisor:** Prof. Tay Wee Peng  
**Institution:** Nanyang Technological University, Singapore  
**MSc Thesis:** School of Electrical and Electronic Engineering

---

## 📖 Overview

This repository contains the official implementation of **Semantic-Guided 3D Gaussian Splatting (3DGS)**, a novel approach that integrates **Segment Anything Model (SAM)** and **CLIP** into 3D Gaussian Splatting to achieve high-quality indoor scene reconstruction under GPU memory constraints.

### Key Features

- **🎯 23.5% Average LPIPS Improvement** on indoor scenes
- **💾 2.3-2.9× Memory Reduction** (1.1M vs 2.5-3.2M Gaussians)
- **⚡ Real-time Rendering** at 25+ FPS with enhanced perceptual quality
- **🏆 Up to 53% LPIPS Boost** on object-dense scenes (Playground)

---

## 🚀 Method Overview

Our approach introduces semantic guidance into 3DGS through three core modules:

### 1. **SAM-SI** (Semantic Sparse Initialization)
- Integrates SAM segmentation with CLIP classification
- Computes per-Gaussian semantic importance scores
- Enhances initialization in perceptually critical regions

### 2. **SAM-RPS** (Semantic-Aware Adaptive Densification)
- Biases densification toward semantically important regions
- Lowers gradient threshold by 30% for high-importance Gaussians (SI > 0.6)
- Memory-constrained mode at 1.1M Gaussian limit

### 3. **Training Pipeline** (`train_semantic.py`)
- Integrated semantic-guided training workflow
- 0.5× resolution training with full-resolution evaluation
- Scene-specific semantic configurations
- Memory monitoring and stability tracking

---

## 📊 Results

### Indoor Scenes (Mip-NeRF 360 Dataset)

| Scene | Baseline LPIPS | Ours LPIPS | Improvement |
|-------|----------------|------------|-------------|
| **Playground** | 0.363 | 0.170 | **+53.2%** |
| **Counter** | 0.286 | 0.191 | **+33.2%** |
| **Room** | 0.339 | 0.267 | **+21.2%** |
| **Kitchen** | 0.203 | 0.169 | **+16.7%** |

**Average LPIPS Improvement: 23.5%**

### Outdoor Scenes (Mip-NeRF 360 & Tanks & Temples)

Achieves **stable training** with 1.1M Gaussians vs baseline requiring 2.5-3.2M Gaussians (**2.3-2.9× memory reduction**).

### Regional Quality Analysis

- **Semantic Objects:** +1.04dB PSNR, +16.6% LPIPS vs backgrounds
- **Trade-off:** Sacrifices background quality for foreground detail

---

## 🛠️ Installation

### Prerequisites
- Linux (tested on Ubuntu 20.04)
- CUDA 11.8+
- Python 3.9+
- 12GB+ GPU memory (RTX 3060 or better)

### Step 1: Clone Repository
```bash
git clone https://github.com/Eureka7771/semantic-3dgs-github.git
cd semantic-3dgs-github
git submodule update --init --recursive
```

### Step 2: Create Conda Environment
```bash
conda env create -f environment.yml
conda activate semantic_3dgs
```

### Step 3: Install Submodules
```bash
# Install diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization

# Install simple-knn
pip install submodules/simple-knn
```

### Step 4: Download SAM Checkpoint
```bash
mkdir -p models
cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

---

## 📦 Dataset Preparation

### Mip-NeRF 360 Dataset
```bash
# Download from https://jonbarron.info/mipnerf360/
mkdir -p data
cd data
# Extract scenes: kitchen, room, counter, etc.
```

### Directory Structure
```
data/
├── kitchen/
│   ├── images/
│   ├── sparse/
│   └── ...
├── room/
└── ...
```

---

## 🎯 Quick Start

### 1. Preprocess SAM Masks (Optional but Recommended)
```bash
python sam_preprocess.py \
    --source_path data/kitchen \
    --output_path data/kitchen/sam_preprocessed \
    --sam_checkpoint models/sam_vit_h_4b8939.pth \
    --quality high
```

**Time:** ~20-40 minutes per scene (one-time preprocessing)

### 2. Train with Semantic Guidance
```bash
python train_semantic.py \
    -s data/kitchen \
    -m output/kitchen \
    --sam_checkpoint models/sam_vit_h_4b8939.pth \
    --use_preprocessed_masks \
    --preprocessed_masks_dir data/kitchen/sam_preprocessed \
    --eval
```

**Training Time:** ~30 minutes (30k iterations on RTX 3060)

### 3. Render Results
```bash
python render.py \
    -m output/kitchen \
    --skip_train \
    --skip_test
```

### 4. Evaluate Metrics
```bash
python metrics.py \
    -m output/kitchen
```

---

## ⚙️ Configuration

### Scene-Specific Semantic Labels

Edit `train_semantic.py` to customize semantic labels for your scene:
```python
# Example: Kitchen scene
scene_specific_labels = {
    'kitchen': ['cabinet', 'counter', 'appliance', 'utensil', 'furniture']
}
```

### Key Training Parameters
```bash
--iterations 30000          # Total training iterations
--densify_from_iter 500    # Start densification
--densify_until_iter 15000 # Stop densification
--densification_interval 100  # Densification frequency
--semantic_weight 1.5      # Semantic importance weight
--max_gaussians 1100000    # Memory limit (1.1M Gaussians)
```

---

## 📁 Repository Structure
```
semantic-3dgs-github/
├── scene/
│   ├── gaussian_model.py          # Modified Gaussian model with semantic attributes
│   ├── cameras.py                 # Camera system
│   └── dataset_readers.py         # COLMAP data loading
├── semantic_3dgs/
│   ├── core/
│   │   ├── adaptive_densification.py  # SAM-RPS module
│   │   └── semantic_initializer.py    # SAM-SI module
│   ├── trainer/
│   │   └── semantic_gaussian_trainer.py  # Training orchestrator
│   ├── models/                    # SAM & CLIP integration
│   └── utils/                     # Visualization utilities
├── utils/                         # 3DGS core utilities
├── gaussian_renderer/             # CUDA rasterization
├── sam_preprocess.py              # SAM offline preprocessing
├── train_semantic.py              # Main training script
├── render.py                      # Rendering script
├── metrics.py                     # Evaluation metrics
└── configs/                       # Configuration files
```

---

## 🔬 Ablation Studies

Our experiments demonstrate:

1. **Semantic Label Quality is Critical**
   - Correct labels: +23-53% LPIPS improvement
   - Incorrect labels: -27% to -63% degradation

2. **Method Excels on Object-Dense Indoor Scenes**
   - Playground (+53%), Counter (+33%)
   - Focuses reconstruction on perceptually important regions

3. **Outdoor Scene Memory Efficiency**
   - Enables stable training with 1.1M Gaussians
   - Baseline requires 2.5-3.2M Gaussians

---

## 📝 Citation

If you find this work useful, please cite:
```bibtex
@mastersthesis{huang2025semantic3dgs,
  author    = {Huang, Yucheng},
  title     = {Semantic-Guided 3D Gaussian Splatting for Resource-Efficient Indoor Scene Reconstruction},
  school    = {Nanyang Technological University},
  year      = {2025},
  type      = {MSc Thesis},
  address   = {Singapore}
}
```

---

## 🙏 Acknowledgements

- **Supervisor:** Prof. Tay Wee Peng for guidance and support
- **Original 3DGS:** [Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **SAM:** [Kirillov et al., Meta AI](https://github.com/facebookresearch/segment-anything)
- **CLIP:** [Radford et al., OpenAI](https://github.com/openai/CLIP)

---

## 📧 Contact

**Huang Yucheng**  
Nanyang Technological University  
GitHub: [@Eureka7771](https://github.com/Eureka7771)

---

## 📄 License

This project is built upon the original 3D Gaussian Splatting codebase. Please refer to [LICENSE.md](LICENSE.md) for details.
