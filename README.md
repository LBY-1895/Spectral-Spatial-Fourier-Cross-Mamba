# Spectral-Spatial-Fourier-Cross-Mamba

Spectral-Spatial-Fourier Cross Mamba (SSFCMamba) is a novel deep learning network designed to address information loss, data redundancy, noise introduction, and insufficient fusion issues in land cover classification (LCC) with hyperspectral imaging and LiDAR data. 

## Overview

...

## Repository Structure

```plain
Spectral-Spatial-Fourier-Cross-Mamba/
├── dataset/               # Dataset storage (put HSI datasets here)
├── .gitattributes         # Git LFS configuration for large files(datasets)
├── demo.py                # Quick start demo for inference
├── loadDat.py             # Dataset loading and preprocessing utilities
├── mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl  # Mamba SSM wheel (CUDA 11.8, PyTorch 2.0)
├── model-90houston.pth    # Pretrained model on Houston dataset (90%+ accuracy)
├── module.py              # Core model components (Spectral-Spatial blocks, Fourier layers, Mamba modules)
├── requirements.txt       # Dependencies list
├── structure.py           # Full model architecture definition
├── test.py                # Evaluation script for model testing
└── utils.py               # Helper functions (metrics, visualization, logging)
```

