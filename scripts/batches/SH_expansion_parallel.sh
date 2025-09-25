#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --tmp=50G                        # per node!!
#SBATCH --job-name=SH_expansion
#SBATCH --output=SH_expansion.out
#SBATCH --error=SH_expansion.err
#SBATCH --mail-type=END

source ../../../my_venv/bin/activate

module load stack/.2024-06-silent
module load gcc/12.2.0
module load python/3.11.6

cd ../../
python scripts/SH_expansion_parallel.py


