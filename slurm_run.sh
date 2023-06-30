#!/bin/bash

#SBATCH --job-name=wind_gpu_test             # Job name
#SBATCH --partition=ais-gpu                   # Queue name 
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=v.morozov@skoltech.ru   # Where to send mail
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=4                     
#SBATCH --cpus-per-task=8     
#SBATCH --gpus-per-task=1
#SBATCH --mem=400G   

#SBATCH --time=48:00:00                      # Time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/gpfs/gpfs0/v.morozov/parallel_%j.log   


srun singularity exec --nv wind_container_p.simg /opt/conda/bin/python src/regression/train.py 