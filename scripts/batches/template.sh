#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=50G                        # per node!!
#SBATCH --mail-type=END
#SBATCH --job-name=template

source ../../../my_venv/bin/activate

module load stack/.2024-06-silent
module load gcc/12.2.0
module load python/3.11.6

python ../template.py --alpha "$1"


