#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon-p8
#SBATCH -o slurm_mc_out/mc_%A_%a.out
#SBATCH -J mc_polymer
#SBATCH --array=0-159%159

source /etc/profile
module load anaconda/2023a-tensorflow


mkdir -p mc_output slurm_mc_out

python mc.py --task-id ${SLURM_ARRAY_TASK_ID}
