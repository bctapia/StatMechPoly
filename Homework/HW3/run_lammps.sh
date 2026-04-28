#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=xeon-p8
#SBATCH -o output_%j.out

# Load environment and modules
source /etc/profile
module load anaconda/2023a-tensorflow
module load intel/oneapi/compiler/latest
module load intel/oneapi/mpi/latest
module load intel/oneapi/mkl/latest

eval "$(conda shell.bash hook)"
conda activate lammps_test_env

# Set environment variables
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Define LAMMPS path
LAMMPS_USE="$HOME/lammps-install/bin/lmp"

# Run LAMMPS using conda and SLURM integration
mpirun -n 1 $LAMMPS_USE -in in.polymer_md