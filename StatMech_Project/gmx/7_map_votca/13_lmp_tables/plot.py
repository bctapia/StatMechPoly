import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def read_votca_bond(file):
    rows = []

    with open(file) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue

            parts = line.split()
            if not parts:
                continue

            try:
                r_nm = float(parts[0])
                U_kj = float(parts[-1])
            except ValueError:
                continue

            rows.append([r_nm, U_kj])

    data = np.array(rows)

    # nm -> Å
    r_ang = data[:, 0] * 10.0
    U_kj = data[:, 1]

    return r_ang, U_kj


def read_lammps_bond(file):
    rows = []
    in_table = False

    with open(file) as f:
        for line in f:
            parts = line.split()

            if len(parts) == 2 and parts[0] == "N":
                in_table = True
                continue

            if not in_table or len(parts) != 4:
                continue

            try:
                idx = int(parts[0])
                r_ang = float(parts[1])
                U_kcal = float(parts[2])
            except ValueError:
                continue

            rows.append([idx, r_ang, U_kcal])

    data = np.array(rows)

    r_ang = data[:, 1]

    # kcal/mol -> kJ/mol
    U_kj = data[:, 2] * 4.184

    return r_ang, U_kj


def plot_bonds():
    lammps_dir = Path(__file__).resolve().parent
    votca_dir = (
        lammps_dir.parent.parent.parent
        / "analyze"
        / "1_map_votca"
        / "bi"
    )

    votca_files = sorted(votca_dir.glob("bond_*.dist.new"))

    plot_data = []

    for votca_file in votca_files:

        # bond_c5-c5.dist.new -> bond_c5_c5.table
        stem = votca_file.stem.replace(".dist", "")
        lammps_name = stem.replace("-", "_") + ".table"
        lammps_file = lammps_dir / lammps_name

        if not lammps_file.exists():
            print(f"Skipping {votca_file.name}: missing {lammps_file.name}")
            continue

        xvot, yvot = read_votca_bond(votca_file)
        xlmp, ylmp = read_lammps_bond(lammps_file)

        plot_data.append((stem, xvot, yvot, xlmp, ylmp))

    n = len(plot_data)

    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, (name, xvot, yvot, xlmp, ylmp) in zip(axes, plot_data):

        ax.plot(
            xvot,
            yvot,
            label="VOTCA",
            marker="o",
            markersize=2,
            linewidth=1
        )

        ax.plot(
            xlmp,
            ylmp,
            label="LAMMPS",
            linewidth=2
        )

        ax.set_title(name)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("U (kJ/mol)")
        ax.set_xlim(np.min(xlmp), np.max(xlmp))
        ax.legend()

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def read_votca_angle(file):
    rows = []
    with open(file) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue

            parts = line.split()
            if not parts:
                continue

            try:
                x = float(parts[0])
                U = float(parts[-1])
            except ValueError:
                continue

            rows.append([x, U])

    data = np.array(rows)
    x_deg = np.rad2deg(data[:, 0])
    U_kj = data[:, 1]
    return x_deg, U_kj


def read_lammps_angle(file):
    rows = []
    in_table = False

    with open(file) as f:
        for line in f:
            parts = line.split()

            if len(parts) == 2 and parts[0] == "N":
                in_table = True
                continue

            if not in_table or len(parts) != 4:
                continue

            try:
                idx = int(parts[0])
                theta = float(parts[1])
                U = float(parts[2])
            except ValueError:
                continue

            rows.append([idx, theta, U])

    data = np.array(rows)
    x_deg = data[:, 1]
    U_kj = data[:, 2] * 4.184
    return x_deg, U_kj


def plot_angles():
    lammps_dir = Path(__file__).resolve().parent
    votca_dir = (
        lammps_dir.parent.parent.parent
        / "analyze"
        / "1_map_votca"
        / "bi"
    )

    votca_files = sorted(votca_dir.glob("angle_*.dist.new"))

    plot_data = []

    for votca_file in votca_files:

        # angle_c5-c5-c5.dist.new -> angle_c5_c5_c5.table
        stem = votca_file.stem.replace(".dist", "")
        lammps_name = stem.replace("-", "_") + ".table"
        lammps_file = lammps_dir / lammps_name

        if not lammps_file.exists():
            print(f"Skipping {votca_file.name}: missing {lammps_file.name}")
            continue

        xvot, yvot = read_votca_angle(votca_file)
        xlmp, ylmp = read_lammps_angle(lammps_file)

        plot_data.append((stem, xvot, yvot, xlmp, ylmp))

    n = len(plot_data)

    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, (name, xvot, yvot, xlmp, ylmp) in zip(axes, plot_data):

        ax.plot(
            xvot,
            yvot,
            label="VOTCA",
            marker="o",
            markersize=2,
            linewidth=1
        )

        ax.plot(
            xlmp,
            ylmp,
            label="LAMMPS",
            linewidth=2
        )

        ax.set_title(name)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("U (kJ/mol)")
        ax.set_xlim(np.min(xlmp), np.max(xlmp))
        ax.legend()

    # hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def read_votca_dihedral(file):
    rows = []

    with open(file) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue

            parts = line.split()
            if not parts:
                continue

            try:
                phi_rad = float(parts[0])
                U_kj = float(parts[-1])
            except ValueError:
                continue

            rows.append([phi_rad, U_kj])

    data = np.array(rows)

    phi_deg = np.rad2deg(data[:, 0])
    U_kj = data[:, 1]

    return phi_deg, U_kj


def read_lammps_dihedral(file):
    rows = []
    in_table = False

    with open(file) as f:
        for line in f:
            parts = line.split()

            if len(parts) == 2 and parts[0] == "N":
                in_table = True
                continue

            if not in_table or len(parts) != 4:
                continue

            try:
                idx = int(parts[0])
                phi_deg = float(parts[1])
                U_kcal = float(parts[2])
            except ValueError:
                continue

            rows.append([idx, phi_deg, U_kcal])

    data = np.array(rows)

    phi_deg = data[:, 1]
    U_kj = data[:, 2] * 4.184

    return phi_deg, U_kj


def plot_dihedrals():
    lammps_dir = Path(__file__).resolve().parent
    votca_dir = (
        lammps_dir.parent.parent.parent
        / "analyze"
        / "1_map_votca"
        / "bi"
    )

    votca_files = sorted(votca_dir.glob("dihedral_*.dist.new"))

    plot_data = []

    for votca_file in votca_files:
        # dihedral_c5-c5-c5-c5.dist.new -> dihedral_c5_c5_c5_c5.table
        stem = votca_file.stem.replace(".dist", "")
        lammps_name = stem.replace("-", "_") + ".table"
        lammps_file = lammps_dir / lammps_name

        if not lammps_file.exists():
            print(f"Skipping {votca_file.name}: missing {lammps_file.name}")
            continue

        xvot, yvot = read_votca_dihedral(votca_file)
        xlmp, ylmp = read_lammps_dihedral(lammps_file)

        plot_data.append((stem, xvot, yvot, xlmp, ylmp))

    n = len(plot_data)

    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, (name, xvot, yvot, xlmp, ylmp) in zip(axes, plot_data):
        ax.plot(
            xvot,
            yvot,
            label="VOTCA",
            marker="o",
            markersize=2,
            linewidth=1,
        )

        ax.plot(
            xlmp,
            ylmp,
            label="LAMMPS",
            linewidth=2,
        )

        ax.set_title(name)
        ax.set_xlabel("phi (deg)")
        ax.set_ylabel("U (kJ/mol)")
        ax.set_xlim(np.min(xlmp), np.max(xlmp))
        ax.legend()

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def read_votca_pair(file):
    rows = []

    with open(file) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue

            parts = line.split()
            if not parts:
                continue

            try:
                r_nm = float(parts[0])
                U_kj = float(parts[-1])
            except ValueError:
                continue

            rows.append([r_nm, U_kj])

    data = np.array(rows)

    r_ang = data[:, 0] * 10.0
    U_kj = data[:, 1]

    return r_ang, U_kj


def read_lammps_pair(file):
    rows = []
    in_table = False

    with open(file) as f:
        for line in f:
            parts = line.split()

            if parts and parts[0] == "N":
                in_table = True
                continue

            if not in_table or len(parts) < 3:
                continue

            try:
                idx = int(parts[0])
                r_ang = float(parts[1])
                U_kcal = float(parts[2])
            except ValueError:
                continue

            rows.append([idx, r_ang, U_kcal])

    if not rows:
        raise ValueError(f"No numeric rows found in {file}")

    data = np.array(rows)

    r_ang = data[:, 1]
    U_kj = data[:, 2] * 4.184

    return r_ang, U_kj


def plot_pairs():
    lammps_dir = Path(__file__).resolve().parent
    votca_dir = (
        lammps_dir.parent.parent.parent
        / "analyze"
        / "7_map_votca"
        / "fm"
    )

    # VOTCA nonbonded files are usually nb_*.dist.new
    votca_files = sorted(votca_dir.glob("nb_*.force"))

    plot_data = []

    for votca_file in votca_files:
        # nb_c5-c5.dist.new -> pair_c5_c5.table
        stem = votca_file.stem #.replace(".dist", "")
        lammps_stem = stem.replace("-", "_")
        lammps_file = lammps_dir / f"{lammps_stem}.table"

        if not lammps_file.exists():
            print(f"Skipping {votca_file.name}: missing {lammps_file.name}")
            continue

        xvot, yvot = read_votca_pair(votca_file)
        xlmp, ylmp = read_lammps_pair(lammps_file)

        plot_data.append((stem, xvot, yvot, xlmp, ylmp))

    n = len(plot_data)

    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, (name, xvot, yvot, xlmp, ylmp) in zip(axes, plot_data):
        ax.plot(
            xvot,
            yvot,
            label="VOTCA",
            marker="o",
            markersize=2,
            linewidth=1,
        )

        ax.plot(
            xlmp,
            ylmp,
            label="LAMMPS",
            linewidth=2,
        )

        ax.set_title(name)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("U (kJ/mol)")
        ax.set_xlim(0, np.max(xlmp))
        ax.set_ylim(np.min(ylmp), 50)
        ax.legend()

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

#plot_bonds()
#plot_angles()
#plot_dihedrals()
plot_pairs()