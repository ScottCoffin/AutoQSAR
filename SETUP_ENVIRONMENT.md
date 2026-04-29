# AutoQSAR Environment Setup

This script helps you set up the conda environment for AutoQSAR based on your system's capabilities.

## Quick Start

### For CUDA-enabled GPU systems:
```powershell
conda env create -f environment-cuda.yml -n autoqsar-py311
conda activate autoqsar-py311
```

### For CPU-only systems:
```powershell
conda env create -f environment-cpu.yml -n autoqsar-py311
conda activate autoqsar-py311
```

## Environment Files Explained

| File | Best For | PyTorch Variant |
|------|----------|-----------------|
| `environment-cuda.yml` | Systems with NVIDIA GPU + CUDA installed | CUDA 12.1 + cuDNN 9 |
| `environment-cpu.yml` | Laptops or systems without GPU | CPU-only (MKL-optimized) |
| `environment-base.yml` | Manual/custom PyTorch installation | None (omitted) |

## How to Choose

- **Have NVIDIA GPU with CUDA?** → Use `environment-cuda.yml`
- **CPU-only machine?** → Use `environment-cpu.yml`
- **Want custom PyTorch?** → Use `environment-base.yml` then manually install PyTorch

## Verification

After activation, verify PyTorch installation:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should be True for CUDA, False for CPU
```

## Common Issues

### ImportError: DLL load failed
- CUDA version mismatch. Ensure CUDA 12.1+ is installed on your system
- Try CPU version instead: `environment-cpu.yml`

### ModuleNotFoundError: No module named 'torch'
- Make sure conda environment is activated: `conda activate autoqsar-py311`

### Environment creation fails
- Try clearing pip cache: `pip cache purge`
- Then retry: `conda env create -f environment-cuda.yml -n autoqsar-py311`
