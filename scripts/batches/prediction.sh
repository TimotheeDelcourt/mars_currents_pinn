#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:20g
#SBATCH --cpus-per-task=8
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=50G                        # per node!!
#SBATCH --mail-type=END
#SBATCH --job-name=template

source ../../../my_venv/bin/activate

module load stack/.2024-06-silent
module load gcc/12.2.0
module load python/3.11.6

python ../prediction.py