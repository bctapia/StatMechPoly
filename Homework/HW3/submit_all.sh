#!/bin/bash

RUN_SCRIPT="run_lammps.sh"

for dir in lammps_runs/*; do
    if [ -d "$dir" ]; then
        echo "Preparing $dir"

        # Copy the SLURM script into the directory
        cp "$RUN_SCRIPT" "$dir/"

        echo "Submitting $dir"
        (cd "$dir" && sbatch run_lammps.sh)
    fi
done