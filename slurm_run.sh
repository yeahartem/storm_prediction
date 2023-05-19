#!/bin/bash
#SBATCH -J singularity
#SBATCH -o singularity.out
#SBATCH -e singularity.err
#SBATCH -p gpu_devel
#SBATCH -t 0-12:00
#SBATCH -c 4
#SBATCH --mem=32000
export WANDB_API_KEY=42
export WANDB_DATA_DIR=/app/wind/outputs
export WANDB_DIR=/app/wind/outputs
export WANDB_CACHE_DIR=/app/wind/outputs

singularity run --bind $(pwd)/Wind:/app/wind --bind $(pwd)/data:/app/wind/data_mounted --pwd "/app/wind" docker://teshbek/xtr_weather:wind_dev116 pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && pip install -r $(pwd)/Wind/environments/pytorch1.13.1_cuda116/requirements_nv.txt && python3 Wind/src/regression/train.py
