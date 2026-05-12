#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon-p8
#SBATCH --array=0-15
#SBATCH -o slurm_md_out/output_%A_%a.out

# Load environment and modules
source /etc/profile

mkdir -p slurm_md_out
mkdir -p md_data
python md_double_well.py
