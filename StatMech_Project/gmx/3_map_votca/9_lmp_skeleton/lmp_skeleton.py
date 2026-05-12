from pathlib import Path
import re


def read_gro(gro_file):
    gro_file = Path(gro_file)
    lines = gro_file.read_text().splitlines()

    natoms = int(lines[1].strip())
    atom_lines = lines[2:2 + natoms]
    box = [float(x) for x in lines[2 + natoms].split()[:3]]

    atoms = {}

    for line in atom_lines:
        atom_id = int(line[15:20])
        atom_name = line[10:15].strip()
        res_id = int(line[0:5])
        res_name = line[5:10].strip()

        x = float(line[20:28]) * 10.0
        y = float(line[28:36]) * 10.0
        z = float(line[36:44]) * 10.0

        atoms[atom_id] = {
            "atom_name": atom_name,
            "res_id": res_id,
            "res_name": res_name,
            "x": x,
            "y": y,
            "z": z,
        }

    return atoms, [b * 10.0 for b in box]


def normalize_header(line):
    m = re.match(r"\s*\[\s*(\w+)\s*\]", line)
    if m:
        return m.group(1).lower()
    return None


def parse_gromacs_top(top_file):
    top_file = Path(top_file)
    lines = top_file.read_text().splitlines()

    sections = {
        "atoms": [],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
    }

    current = None

    for line in lines:
        header = normalize_header(line)
        if header is not None:
            current = header
            continue

        if current not in sections:
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        sections[current].append(stripped)

    atoms = {}
    bead_type_to_id = {}

    for line in sections["atoms"]:
        main = line.split(";")[0].split()
        atom_id = int(main[0])
        bead_type = main[1]
        resnr = int(main[2])
        resid = main[3]
        atom_name = main[4]
        cgnr = int(main[5])
        charge = float(main[6])
        mass = float(main[7])

        if bead_type not in bead_type_to_id:
            bead_type_to_id[bead_type] = len(bead_type_to_id) + 1

        atoms[atom_id] = {
            "bead_type": bead_type,
            "type_id": bead_type_to_id[bead_type],
            "resnr": resnr,
            "resid": resid,
            "atom_name": atom_name,
            "cgnr": cgnr,
            "charge": charge,
            "mass": mass,
        }

    def parse_interactions(section, n_atoms):
        out = []
        name_to_id = {}

        for line in sections[section]:
            main, *comment = line.split(";")
            cols = main.split()

            atom_ids = tuple(int(x) for x in cols[:n_atoms])

            comment_text = comment[0].strip() if comment else ""
            m = re.search(r"\d+:([^:]+):\d+", comment_text)

            if m:
                interaction_name = m.group(1)
            else:
                interaction_name = section[:-1]

            if interaction_name not in name_to_id:
                name_to_id[interaction_name] = len(name_to_id) + 1

            out.append(
                {
                    "atoms": atom_ids,
                    "type_name": interaction_name,
                    "type_id": name_to_id[interaction_name],
                }
            )

        return out, name_to_id

    bonds, bond_types = parse_interactions("bonds", 2)
    angles, angle_types = parse_interactions("angles", 3)
    dihedrals, dihedral_types = parse_interactions("dihedrals", 4)

    return {
        "atoms": atoms,
        "bead_type_to_id": bead_type_to_id,
        "bonds": bonds,
        "bond_types": bond_types,
        "angles": angles,
        "angle_types": angle_types,
        "dihedrals": dihedrals,
        "dihedral_types": dihedral_types,
    }


def write_lammps_skeleton(gro_file, top_file, out_file):
    gro_atoms, box = read_gro(gro_file)
    top = parse_gromacs_top(top_file)

    atoms = top["atoms"]

    missing = sorted(set(atoms) - set(gro_atoms))
    if missing:
        raise ValueError(f"{len(missing)} topology atoms missing from gro. First few: {missing[:10]}")

    lx, ly, lz = box

    out_file = Path(out_file)

    with out_file.open("w", newline="\n") as f:
        f.write("UA skeleton generated from GROMACS ua.gro + ua_topol.top\n\n")

        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(top['bonds'])} bonds\n")
        f.write(f"{len(top['angles'])} angles\n")
        f.write(f"{len(top['dihedrals'])} dihedrals\n")
        f.write("0 impropers\n\n")

        f.write(f"{len(top['bead_type_to_id'])} atom types\n")
        f.write(f"{len(top['bond_types'])} bond types\n")
        f.write(f"{len(top['angle_types'])} angle types\n")
        f.write(f"{len(top['dihedral_types'])} dihedral types\n")
        f.write("0 improper types\n\n")

        f.write(f"{0.0:.8f} {lx:.8f} xlo xhi\n")
        f.write(f"{0.0:.8f} {ly:.8f} ylo yhi\n")
        f.write(f"{0.0:.8f} {lz:.8f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for bead_type, type_id in sorted(top["bead_type_to_id"].items(), key=lambda x: x[1]):
            mass = next(a["mass"] for a in atoms.values() if a["bead_type"] == bead_type)
            f.write(f"{type_id} {mass:.6f} # {bead_type}\n")

        f.write("\nAtoms # full\n\n")
        for atom_id in sorted(atoms):
            a = atoms[atom_id]
            g = gro_atoms[atom_id]

            mol_id = 1
            f.write(
                f"{atom_id} {mol_id} {a['type_id']} {a['charge']:.8f} "
                f"{g['x']:.8f} {g['y']:.8f} {g['z']:.8f} "
                f"# {a['bead_type']} {a['atom_name']}\n"
            )

        f.write("\nBonds\n\n")
        for i, b in enumerate(top["bonds"], start=1):
            a1, a2 = b["atoms"]
            f.write(f"{i} {b['type_id']} {a1} {a2} # {b['type_name']}\n")

        f.write("\nAngles\n\n")
        for i, ang in enumerate(top["angles"], start=1):
            a1, a2, a3 = ang["atoms"]
            f.write(f"{i} {ang['type_id']} {a1} {a2} {a3} # {ang['type_name']}\n")

        f.write("\nDihedrals\n\n")
        for i, dih in enumerate(top["dihedrals"], start=1):
            a1, a2, a3, a4 = dih["atoms"]
            f.write(f"{i} {dih['type_id']} {a1} {a2} {a3} {a4} # {dih['type_name']}\n")


if __name__ == "__main__":
    write_lammps_skeleton(
        gro_file="ua.gro",
        top_file="ua_topol.top",
        out_file="ua_skeleton.data",
    )