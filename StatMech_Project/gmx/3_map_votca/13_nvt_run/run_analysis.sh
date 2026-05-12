#!/bin/bash
#SBATCH -J c2bond
#SBATCH -N 1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -p xeon-p8
#SBATCH -o analysis_out.out

set -eo pipefail

module purge
module load anaconda/2023b

# Initialize conda for *this* non-interactive shell (critical)
eval "$(/state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/bin/conda shell.bash hook)"

conda activate aging

echo "Host: $(hostname)"

python -u analyze.py

