import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path


def parse_votca_atom_token(token):
    """
    VOTCA token format in your mapping:
        1:R01:A1a

    Atom name is A<hex_id>, so:
        A1  -> 1
        Aa  -> 10
        A10 -> 16
        A1a -> 26
    """
    atom_name = token.split(":")[2]
    return int(atom_name[1:], 16)


def get_section(line):
    s = line.strip()
    if s.startswith("[") and s.endswith("]"):
        return s.strip("[]").strip().lower()
    return None


def make_nb_exclusion_top(
    mapping_xml,
    top_in,
    top_out="nb_exclusion.top",
    exclude_13=True,
):
    mapping_xml = Path(mapping_xml)
    top_in = Path(top_in)
    top_out = Path(top_out)

    # ------------------------------------------------------------
    # 1. Parse CG bead membership from mapping.xml
    # ------------------------------------------------------------
    tree = ET.parse(mapping_xml)
    root = tree.getroot()

    bead_atoms = {}

    for bead in root.findall(".//cg_bead"):
        name = bead.findtext("name")
        beads = bead.findtext("beads")

        if name is None or beads is None:
            continue

        name = name.strip()
        atom_ids = [parse_votca_atom_token(tok) for tok in beads.split()]
        bead_atoms[name] = atom_ids

    print(f"Parsed {len(bead_atoms)} CG beads")

    # ------------------------------------------------------------
    # 2. Build AA atom -> CG bead map
    # ------------------------------------------------------------
    atom_to_bead = {}

    for bead_name, atoms in bead_atoms.items():
        for atom in atoms:
            if atom in atom_to_bead:
                raise ValueError(
                    f"Atom {atom} appears in multiple CG beads: "
                    f"{atom_to_bead[atom]} and {bead_name}"
                )
            atom_to_bead[atom] = bead_name

    # ------------------------------------------------------------
    # 3. Infer CG bonds from AA [ bonds ] section in topology
    # ------------------------------------------------------------
    top_lines = top_in.read_text().splitlines()

    cg_bonds = set()
    current_section = None

    for line in top_lines:
        sec = get_section(line)
        if sec is not None:
            current_section = sec
            continue

        stripped = line.strip()

        if current_section != "bonds":
            continue

        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue

        body = line.split(";", 1)[0]
        fields = body.split()

        if len(fields) < 2:
            continue

        try:
            i = int(fields[0])
            j = int(fields[1])
        except ValueError:
            continue

        bead_i = atom_to_bead.get(i)
        bead_j = atom_to_bead.get(j)

        if bead_i is None or bead_j is None:
            continue

        if bead_i != bead_j:
            cg_bonds.add(tuple(sorted((bead_i, bead_j))))

    print(f"Inferred {len(cg_bonds)} CG bonds from AA topology")

    # ------------------------------------------------------------
    # 4. Build CG graph and collect 1-2 / 1-3 CG exclusions
    # ------------------------------------------------------------
    graph = defaultdict(set)

    for a, b in cg_bonds:
        graph[a].add(b)
        graph[b].add(a)

    excluded_cg_pairs = set(cg_bonds)

    if exclude_13:
        for center, neighbors in graph.items():
            for a, b in combinations(sorted(neighbors), 2):
                excluded_cg_pairs.add(tuple(sorted((a, b))))

    print(f"Total CG excluded pairs: {len(excluded_cg_pairs)}")

    # ------------------------------------------------------------
    # 5. Expand CG exclusions into AA atom exclusions
    # ------------------------------------------------------------
    aa_exclusions = defaultdict(set)

    # Exclude atoms between excluded CG bead pairs
    for bead_a, bead_b in excluded_cg_pairs:
        atoms_a = bead_atoms[bead_a]
        atoms_b = bead_atoms[bead_b]

        for i, j in product(atoms_a, atoms_b):
            lo, hi = sorted((i, j))
            aa_exclusions[lo].add(hi)

    # Exclude atoms within the same CG bead
    for atoms in bead_atoms.values():
        for i, j in combinations(sorted(atoms), 2):
            lo, hi = sorted((i, j))
            aa_exclusions[lo].add(hi)

    print(f"AA atoms with exclusions: {len(aa_exclusions)}")

    # ------------------------------------------------------------
    # 6. Zero bonded force constants in topology
    # ------------------------------------------------------------
    bonded_sections = {"bonds", "angles", "dihedrals", "impropers"}
    current_section = None
    out = []

    for line in top_lines:
        sec = get_section(line)
        if sec is not None:
            current_section = sec
            out.append(line)
            continue

        stripped = line.strip()

        if (
            current_section in bonded_sections
            and stripped
            and not stripped.startswith(";")
            and not stripped.startswith("#")
        ):
            body, *comment = line.split(";", 1)
            fields = body.split()

            if current_section == "bonds":
                # ai aj funct r0 k
                if len(fields) >= 5:
                    fields[4] = "0.0"

            elif current_section == "angles":
                # ai aj ak funct theta0 k
                if len(fields) >= 6:
                    fields[5] = "0.0"

            elif current_section in {"dihedrals", "impropers"}:
                # ai aj ak al funct parameters...
                # zero all parameters after funct
                if len(fields) >= 6:
                    for idx in range(5, len(fields)):
                        fields[idx] = "0.0"

            newline = " ".join(fields)
            if comment:
                newline += " ;" + comment[0]

            out.append(newline)

        else:
            out.append(line)

    # ------------------------------------------------------------
    # 7. Append generated [ exclusions ] block
    # ------------------------------------------------------------
    out.append("")
    out.append("[ exclusions ]")
    out.append("; CG 1-2 and 1-3 exclusions generated from mapping.xml + AA bonds")
    out.append("; atoms within the same CG bead are also excluded")

    for i in sorted(aa_exclusions):
        js = sorted(aa_exclusions[i])
        if js:
            out.append(f"{i:8d} " + " ".join(f"{j:8d}" for j in js))

    top_out.write_text("\n".join(out) + "\n")

    print(f"\nWrote {top_out}")


# ---------- RUN ----------
if __name__ == "__main__":
    gmx_dir = Path(__file__).resolve().parent.parent.parent
    votca_dir = Path(__file__).resolve().parent.parent
    make_nb_exclusion_top(
        mapping_xml= votca_dir / "2_mapping" / "mapping.xml",
        top_in= gmx_dir / "nvt_converted_fixed.top",
        top_out= votca_dir / "7_nonbond_AA" / "nb_exclusion.top",
        exclude_13=True,
        )
