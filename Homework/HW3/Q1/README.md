## MD
Run ```make_lammps.py``` to generate the the LAMMPS data files and input files. They are created in the ```lammps_runs/N{number_of_beads}_coil_T{reduced_temperature}``` directories.

Run ```submit_all.sh``` which submits each job using the slurm subject to the ```run_lammps.sh``` submission script.

Run ```analyze_md.py``` to analyze the MD results.


## MC
