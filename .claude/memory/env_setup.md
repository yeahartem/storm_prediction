---
name: Environment setup
description: conda env wind — packages and compatibility fixes needed to run training
type: project
---

**Python env:** `c:\Users\Artem\miniconda3\envs\wind\python.exe` (Python 3.10)

**GPU:** NVIDIA RTX 3070, 8GB VRAM, CUDA 12.7, Driver 566.36

**Key packages (after fixes):**
- torch 2.11.0+cu126 (CUDA-enabled — must force-reinstall if accidentally replaced with CPU)
- pytorch-lightning 2.0.2
- numpy 1.26.4 (must stay <2.0, numpy 2.x breaks wandb and scipy)
- setuptools 69.5.1 (must stay <70 for pkg_resources compatibility with PL 2.0.2)
- wandb 0.25.1 (updated from 0.15.2 which was broken with new protobuf)
- timm 1.0.26, denseweight, mlflow 3.10.1, gitpython

**Run command:** `/c/Users/Artem/miniconda3/envs/wind/python.exe train.py --config-name=cmip6_world`

**Why:** Non-admin account, conda not on PATH in WSL bash. Must use full path.
**How to apply:** Always use full python path; verify CUDA with `torch.cuda.is_available()` before long runs.
