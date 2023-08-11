#!/bin/bash

#SBATCH --job-name=wind_gpu_train             # Job name
#SBATCH --partition=ais-gpu                   # Queue name 
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=v.morozov@skoltech.ru   # Where to send mail
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=8                     
#SBATCH --cpus-per-task=8     
#SBATCH --gpus-per-task=1
#SBATCH --mem=930G   

#SBATCH --time=72:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/trinity/home/v.morozov/logs/parallel_%j.log 
#SBATCH --error=/trinity/home/v.morozov/logs/parallel_error_%j.log 

srun singularity exec --nv wind_container20.simg /opt/conda/bin/python3 src/regression/train.py
