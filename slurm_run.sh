#!/bin/bash

#SBATCH --job-name=wind_gpu_test             # Job name
#SBATCH --partition=gpu                      # Queue name 
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=v.morozov@skoltech.ru   # Where to send mail

#SBATCH --nodes=1                            # Run all processes on a single node

#SBATCH --ntasks=2                           # Run two task

#SBATCH --cpus-per-task=8                    # Number of CPU cores per task

#SBATCH --mem=128000                           # Job memory request in Megabytes

#SBATCH --gpus=2                             # Number of GPUs

#SBATCH --time=00:25:00                      # Time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/gpfs/gpfs0/v.morozov/parallel_%j.log     # Standard output and error log


#MODULE LOAD PART

module load compilers/intel_2018.3.222 gpu/cuda-11.7
export WANDB_API_KEY=7ce4e8a3a21df6f25a3a589a9de3f52c759b3633

singularity run --nv wind_container.simg /opt/conda/bin/python src/regression/train.py 