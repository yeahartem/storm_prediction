#!/bin/bash
#SBATCH -J singularity
#SBATCH -o singularity.out
#SBATCH -e singularity.err
#SBATCH -p gpu_devel
#SBATCH -t 0-12:00
#SBATCH -c 1
#SBATCH --mem=64024
export WANDB_API_KEY=42
export WANDB_DATA_DIR=/app/wind/outputs
export WANDB_DIR=/app/wind/outputs
export WANDB_CACHE_DIR=/app/wind/outputs

singularity run --bind -v $(pwd)/docker_repos/Wind:/app/wind --bind $(pwd)/docker_repos/Wind:/app/wind --pwd "/app/wind" xtr_weather_wind_dev116.sif python3 /app/wind/src/regression/train.py 