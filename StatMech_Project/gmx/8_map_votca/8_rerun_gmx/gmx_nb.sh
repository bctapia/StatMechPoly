#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH --partition=xeon-p8
#SBATCH -o out_%j.out

source /etc/profile
module load anaconda/2023a-tensorflow
module load intel/oneapi/compiler/latest
module load intel/oneapi/mpi/latest
module load intel/oneapi/mkl/latest

eval "$(conda shell.bash hook)"
conda activate gmx2025

GMX_USE=/home/gridsan/btapia/programs/gromacs-cpu/bin/gmx_mpi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=true
export OMP_PLACES=cores

$GMX_USE grompp \
    -f rerun_nb_exclusion.mdp \
    -c nvt_converted_strict.gro \
    -p nb_exclusion.top \
    -o nb_exclusion.tpr

$GMX_USE mdrun \
    -s nb_exclusion.tpr \
    -rerun nvt.trr \
    -deffnm nb_exclusion \
    -ntomp ${OMP_NUM_THREADS} \
    -pin on
