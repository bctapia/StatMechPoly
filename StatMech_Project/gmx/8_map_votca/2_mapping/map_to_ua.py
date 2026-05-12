from collections import defaultdict
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
from pathlib import Path

@dataclass
class Atom:
    idx: int
    label: str
    mass: float

@dataclass
class Bond:
    i: int
    j: int

@dataclass
class Angle:
    i: int
    j: int
    k: int

@dataclass
class Dihedral:
    i: int
    j: int
    k: int
    l: int

def prettify_xml(elem):
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent="  ")

def is_h(atom):
    return abs(atom.mass - 1.008) < 0.2

def parse_csg_dump_atoms(f):
    atoms = []
    r = re.compile(r"^\s*(\d+)\s+Name\s+(\S+).*Mass\s+([0-9eE+\-.]+)")
    for line in open(f):
        m = r.match(line.strip())
        if m:
            atoms.append(Atom(int(m.group(1)), m.group(2), float(m.group(3))))
    return atoms

def parse_section(top, name, n):
    out = []
    active = False
    for line in open(top):
        line = line.strip()
        if line.startswith("["):
            active = name in line
            continue
        if not active or line.startswith(";") or not line:
            continue
        parts = line.split(";")[0].split()
        if len(parts) >= n:
            out.append(tuple(int(x)-1 for x in parts[:n]))
    return out

def parse_top_atoms(top):
    types, names = {}, {}
    active = False
    for line in open(top):
        line = line.strip()
        if line.startswith("["):
            active = "atoms" in line
            continue
        if not active or line.startswith(";") or not line:
            continue
        p = line.split(";")[0].split()
        if len(p) >= 5:
            i = int(p[0]) - 1
            types[i] = p[1]
            names[i] = p[4]
    return types, names

def build_ua_groups(atoms, bonds, types, names, type_map=None):
    neigh = defaultdict(set)
    for i, j in bonds:
        neigh[i].add(j)
        neigh[j].add(i)

    groups = []
    counts = defaultdict(int)

    atom_dict = {a.idx: a for a in atoms}
    for a in atoms:
        if is_h(a):
            continue

        bonded_h = [j for j in neigh[a.idx] if is_h(atom_dict[j])]
        counts[names[a.idx]] += 1

        old_type = types[a.idx]
        new_type = type_map.get(old_type, old_type) if type_map else old_type

        groups.append({
            "center": a.idx,
            "name": f"{names[a.idx]}_{counts[names[a.idx]]}",
            "type": new_type,
            "atoms": [a.idx] + bonded_h,
            "weights": [atoms[i].mass for i in [a.idx] + bonded_h]
        })

    return groups

def canonical(seq):
    return min(tuple(seq), tuple(reversed(seq)))

def build_bead_type_lookup(groups):
    return {g["name"]: g["type"] for g in groups}


def canonical_typed(beads, bead_to_type):
    beads = tuple(beads)
    rev_beads = tuple(reversed(beads))

    types = tuple(bead_to_type[b] for b in beads)
    rev_types = tuple(reversed(types))

    # Canonicalize using types first, then bead names as tie-breaker
    if rev_types < types or (rev_types == types and rev_beads < beads):
        return rev_beads, "-".join(rev_types)

    return beads, "-".join(types)

def build_atom_to_bead(groups):
    atom_to_bead = {}
    for g in groups:
        for i in g["atoms"]:
            atom_to_bead[i] = g["name"]
    return atom_to_bead

def map_terms_by_type(groups, terms, prefix):
    """
    Map AA topology terms to UA bead terms and group by UA type pattern.
    Returns:
        dict[str, list[tuple[str, ...]]]
        e.g. {"lmp_001-lmp_002-lmp_003": [(A1_1, A2_1, A3_1), ...]}
    """
    atom_to_bead = build_atom_to_bead(groups)
    bead_to_type = build_bead_type_lookup(groups)

    grouped = defaultdict(set)

    for term in terms:
        atom_indices = tuple(term)
        bead_seq = tuple(atom_to_bead.get(i) for i in atom_indices)

        if any(b is None for b in bead_seq):
            continue

        # Drop collapsed adjacent interactions, e.g. C-H -> same UA bead
        if any(bead_seq[i] == bead_seq[i + 1] for i in range(len(bead_seq) - 1)):
            continue

        # Drop repeated-bead terms like A-B-A or A-B-C-A
        if len(set(bead_seq)) != len(bead_seq):
            continue

        canon_beads, type_name = canonical_typed(bead_seq, bead_to_type)
        grouped[f"{prefix}_{type_name}"].add(canon_beads)

    return {
        name: sorted(terms)
        for name, terms in sorted(grouped.items())
    }


def write_bonded_blocks(parent, tag, grouped_terms):
    """
    Write one VOTCA bonded block per chemically distinct UA type.
    """
    for type_name, terms in grouped_terms.items():
        if not terms:
            continue

        block = ET.SubElement(parent, tag)
        ET.SubElement(block, "name").text = type_name
        ET.SubElement(block, "beads").text = (
            "\n        "
            + "\n        ".join(" ".join(term) for term in terms)
            + "\n      "
        )

def make_mapping(dump, top, out, ident_name, cg_name, type_map=None):
    atoms = parse_csg_dump_atoms(dump)
    types, names = parse_top_atoms(top)

    bonds = parse_section(top, "bonds", 2)
    angles = parse_section(top, "angles", 3)
    dihs = parse_section(top, "dihedrals", 4)

    groups = build_ua_groups(atoms, bonds, types, names, type_map=type_map)

    cg_bonds = map_terms_by_type(groups, bonds, "bond")
    cg_angles = map_terms_by_type(groups, angles, "angle")
    cg_dihs = map_terms_by_type(groups, dihs, "dihedral")

    root = ET.Element("cg_molecule")
    ET.SubElement(root, "name").text = cg_name
    ET.SubElement(root, "ident").text = ident_name

    top_el = ET.SubElement(root, "topology")
    beads_el = ET.SubElement(top_el, "cg_beads")

    for g in groups:
        b = ET.SubElement(beads_el, "cg_bead")
        ET.SubElement(b, "name").text = g["name"]
        ET.SubElement(b, "type").text = g["type"]
        ET.SubElement(b, "mapping").text = g["name"]
        ET.SubElement(b, "beads").text = " ".join(atoms[i].label for i in g["atoms"])

    bonded = ET.SubElement(top_el, "cg_bonded")
    write_bonded_blocks(bonded, "bond", cg_bonds)
    write_bonded_blocks(bonded, "angle", cg_angles)
    write_bonded_blocks(bonded, "dihedral", cg_dihs)

    maps = ET.SubElement(root, "maps")
    for g in groups:
        m = ET.SubElement(maps, "map")
        ET.SubElement(m, "name").text = g["name"]
        ET.SubElement(m, "weights").text = " ".join(str(w) for w in g["weights"])

    open(out, "w").write(prettify_xml(root))

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    dump_file = base_dir / "2_map_votca" / "1_csg_dump" / "out_4670701.out"
    top_file = base_dir / "nvt_converted_fixed.top"

    type_map = {
        "lmp_001": "c5",
        "lmp_002": "ca",
        "lmp_003": "c3",
        "lmp_004": "c5",
        "lmp_005": "ca",
        "lmp_006": "ca",
        "lmp_007": "c5",
        "lmp_008": "ha",
        "lmp_009": "hc",
    }

    make_mapping(
        dump_file,
        top_file,
        "mapping.xml",
        ident_name="lmp_system",
        cg_name="ua_system",
        type_map=type_map,
    )
