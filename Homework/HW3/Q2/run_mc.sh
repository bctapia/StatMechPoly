#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon-p8
#SBATCH -o output_%j.out

# Load environment and modules
source /etc/profile

python mc_double_well.py
