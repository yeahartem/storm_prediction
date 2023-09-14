#!/bin/bash

#SBATCH --job-name=wind_gpu_train             # Job name
#SBATCH --partition=ais-gpu                   # Queue name 

#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=8                 
#SBATCH --cpus-per-task=8    
#SBATCH --gpus-per-task=1
#SBATCH --mem=0

#SBATCH --time=96:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/trinity/home/v.morozov/logs/parallel_%j.log 
#SBATCH --error=/trinity/home/v.morozov/logs/parallel_error_%j.log 

srun singularity exec --nv xtr_weather_wind_dev2.sif bash << EOF 
/opt/conda/bin/python -m pip install -r requirements.txt
/opt/conda/bin/python src/regression/train.py
EOF