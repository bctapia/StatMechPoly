#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon-p8
#SBATCH -o output_%j.out

# Load environment and modules
source /etc/profile
module load anaconda/2023a-tensorflow
#module load intel/oneapi/compiler/latest
#module load intel/oneapi/mpi/latest
#module load intel/oneapi/mkl/latest

eval "$(conda shell.bash hook)"
conda activate intermol


#python /home/gridsan/btapia/software/InterMol/intermol/convert.py --lmp_in nvt.in --gromacs
# -ls "pair_style cut/coul/long 15"
