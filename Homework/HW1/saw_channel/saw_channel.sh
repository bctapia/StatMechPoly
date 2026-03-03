#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon-p8
#SBATCH -o rw_output_%j.out

. /etc/profile.d/modules.sh

# Load environment and modules
source /etc/profile

eval "$(conda shell.bash hook)"
conda activate base

python -u ../randomwalks.py --mode confined2D
