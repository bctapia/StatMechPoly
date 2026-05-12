#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon-p8
#SBATCH -o out_%j.out

# Load environment and modules
source /etc/profile
module load anaconda/2023a-tensorflow
module load intel/oneapi/mkl/latest
module load mpi/openmpi-5.0.7

eval "$(conda shell.bash hook)"
conda activate votca_build

source /home/gridsan/btapia/software/votca/bin/VOTCARC.bash

csg_dump --top nvt.tpr
