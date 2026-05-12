from pathlib import Path
import itertools


def safe_keyword(name):
    return name


def get_section_type_map(data_file, section, next_section):
    lines = Path(data_file).read_text().splitlines()

    in_section = False
    mapping = {}

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(section):
            in_section = True
            continue

        if next_section is not None and stripped.startswith(next_section):
            in_section = False
            continue

        if not in_section or not stripped or stripped.startswith("#"):
            continue

        main, *comment = stripped.split("#", 1)
        cols = main.split()

        if len(cols) < 2:
            continue

        type_id = int(cols[1])
        type_name = comment[0].strip().replace("-", "_").lower() if comment else None

        if type_name:
            mapping[type_id] = type_name

    return dict(sorted(mapping.items()))


def find_table(table_dir, names):
    for name in names:
        p = Path(table_dir) / f"{name}.table"
        if p.exists():
            return p
    return None


def write_all_coeffs(
    data_file,
    table_dir,
    type_to_name,
    output_file="table_coeffs.in",
):
    table_dir = Path(table_dir)

    bond_map = get_section_type_map(data_file, "Bonds", "Angles")
    angle_map = get_section_type_map(data_file, "Angles", "Dihedrals")
    dihedral_map = get_section_type_map(data_file, "Dihedrals", None)

    with open(output_file, "w", newline="\n") as f:
        f.write("# LAMMPS table coefficients\n\n")

        for type_id, name in bond_map.items():
            table = find_table(table_dir, [name])
            if table is None:
                raise FileNotFoundError(f"Missing bond table for {name}")
            f.write(f"bond_coeff {type_id} {table} {safe_keyword(name)}\n")

        f.write("\n")

        for type_id, name in angle_map.items():
            table = find_table(table_dir, [name])
            if table is None:
                raise FileNotFoundError(f"Missing angle table for {name}")
            f.write(f"angle_coeff {type_id} {table} {safe_keyword(name)}\n")

        f.write("\n")

        for type_id, name in dihedral_map.items():
            table = find_table(table_dir, [name])
            if table is None:
                raise FileNotFoundError(f"Missing dihedral table for {name}")
            f.write(f"dihedral_coeff {type_id} {table} {safe_keyword(name)}\n")

        f.write("\n")

        for i, j in itertools.combinations_with_replacement(sorted(type_to_name), 2):
            a = type_to_name[i]
            b = type_to_name[j]

            candidates = [
                f"nb_{a}_{b}",
                f"nb_{b}_{a}",
                f"pair_{a}_{b}",
                f"pair_{b}_{a}",
                f"nonbond_{a}_{b}",
                f"nonbond_{b}_{a}",
            ]

            table = find_table(table_dir, candidates)
            if table is None:
                raise FileNotFoundError(f"Missing pair table for {i}-{j} ({a}-{b})")

            f.write(f"pair_coeff {i} {j} {table} {safe_keyword(table.stem)}\n")

def get_atom_type_map(data_file):
    lines = Path(data_file).read_text().splitlines()

    in_section = False
    mapping = {}

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Masses"):
            in_section = True
            continue

        if in_section and stripped == "":
            continue

        # next section starts → stop
        if in_section and stripped.startswith("Atoms"):
            break

        if not in_section or stripped.startswith("#") or not stripped:
            continue

        main, *comment = stripped.split("#", 1)
        cols = main.split()

        if len(cols) < 2:
            continue

        type_id = int(cols[0])
        name = comment[0].strip() if comment else None

        if name:
            mapping[type_id] = name

    return dict(sorted(mapping.items()))


if __name__ == "__main__":
    type_to_name = get_atom_type_map(Path(__file__).resolve().parent.parent / "9_lmp_skeleton" /"ua_skeleton.data")
    write_all_coeffs(
        data_file=Path(__file__).resolve().parent.parent / "9_lmp_skeleton" /"ua_skeleton.data",
        table_dir=Path(__file__).resolve().parent.parent / "10_lmp_tables",
        type_to_name=type_to_name,
        output_file="table_coeffs.in",
    )