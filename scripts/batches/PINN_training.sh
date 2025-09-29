#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpumem:20g
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=500G                        # per node!!
#SBATCH --job-name=PINN_training
#SBATCH --mail-type=END

source ../../../my_venv/bin/activate

module load stack/.2024-06-silent
module load gcc/12.2.0
module load python/3.11.6

python ../run_ensemble_training.py


