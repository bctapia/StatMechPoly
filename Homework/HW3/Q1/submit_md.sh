#!/bin/bash

MAX_JOBS=190
RUN_SCRIPT="run_lammps.sh"

dos2unix "$RUN_SCRIPT" submit_md.sh

count_active_jobs() {
    squeue -u "$USER" -h | wc -l
}

for dir in lammps_runs/*; do
    if [ -d "$dir" ]; then

        while [ "$(count_active_jobs)" -ge "$MAX_JOBS" ]; do
            echo "At $MAX_JOBS active jobs. Waiting..."
            sleep 60
        done

        echo "Preparing $dir"
        cp "$RUN_SCRIPT" "$dir/"
        dos2unix "$dir/run_lammps.sh"

        echo "Submitting $dir"
        (cd "$dir" && sbatch run_lammps.sh)

        sleep 1
    fi
done
