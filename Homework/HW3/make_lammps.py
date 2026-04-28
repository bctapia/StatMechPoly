from pathlib import Path
import numpy as np

import mc

def write_lammps_data(r, filename, mass=1.0, box_pad=10.0):
    N = len(r)
    r = r - r.mean(axis=0)

    mins = r.min(axis=0) - box_pad
    maxs = r.max(axis=0) + box_pad

    with open(filename, "w") as f:
        f.write("LAMMPS polymer data\n\n")

        f.write(f"{N} atoms\n")
        f.write(f"{N - 1} bonds\n\n")

        f.write("1 atom types\n")
        f.write("1 bond types\n\n")

        f.write(f"{mins[0]:.8f} {maxs[0]:.8f} xlo xhi\n")
        f.write(f"{mins[1]:.8f} {maxs[1]:.8f} ylo yhi\n")
        f.write(f"{mins[2]:.8f} {maxs[2]:.8f} zlo zhi\n\n")

        f.write("Masses\n\n")
        f.write(f"1 {mass:.8f}\n\n")

        f.write("Atoms\n\n")
        for i, xyz in enumerate(r, start=1):
            f.write(
                f"{i} 1 1 "
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n"
            )

        f.write("\nBonds\n\n")
        for i in range(1, N):
            f.write(f"{i} 1 {i} {i + 1}\n")


def write_lammps_input(
    input_file,
    data_file,
    T,
    n_steps=2_000_000,
    thermo_every=1000,
    dump_every=1000,
    epsilon=1.0,
    sigma=1.0,
    rcut=2.5,
    k_bond=50.0,
    b0=1.0,
    langevin_damp=1.0,
):
    with open(input_file, "w") as f:
        f.write(f"""units lj
atom_style molecular
boundary f f f

read_data {data_file}

pair_style lj/cut {rcut}
pair_coeff 1 1 {epsilon} {sigma} {rcut}

bond_style harmonic
bond_coeff 1 {k_bond} {b0}

special_bonds lj 0.0 0.0 1.0

neighbor 1.0 bin
neigh_modify delay 0 every 1 check yes
comm_modify cutoff 5.0

compute rg all gyration

thermo {thermo_every}
thermo_style custom step temp pe ke etotal c_rg

min_style fire
minimize 1.0e-6 1.0e-8 10000 100000

velocity all create {T} 12345 mom yes rot yes dist gaussian

fix relax all nve/limit 0.01
fix bath all langevin {T} {T} {langevin_damp} 54321 zero yes

timestep 0.001
run 50000

unfix relax
unfix bath

velocity all create {T} 67890 mom yes rot yes dist gaussian

fix int all nve
fix thermostat all langevin {T} {T} {langevin_damp} 98765 zero yes

dump traj all custom {dump_every} traj.lammpstrj id type xu yu zu
dump_modify traj sort id

timestep 0.005
run {n_steps}
""")


def main():
    outdir = Path("lammps_runs")
    outdir.mkdir(exist_ok=True)

    chain_lengths = [50, 100, 150, 200]
    temperatures = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    initial_modes = ["coil"]

    for N in chain_lengths:
        for init in initial_modes:
            for T in temperatures:
                r = mc.make_initial_chain(
                    N=N,
                    b0=1.0,
                    mode=init,
                    seed=1000 + N + int(100 * T),
                )
                run_dir = outdir / f"N{N}_{init}_T{T:.2f}"
                run_dir.mkdir(exist_ok=True)

                data_file = run_dir / "polymer.data"
                input_file = run_dir / "in.polymer_md"

                write_lammps_data(
                    r=r,
                    filename=data_file,
                    mass=1.0,
                    box_pad=10.0,
                )

                write_lammps_input(
                    input_file=input_file,
                    data_file=data_file.name,
                    T=T,
                )

                print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()