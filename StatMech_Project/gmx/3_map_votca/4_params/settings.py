from pathlib import Path
import xml.etree.ElementTree as ET
import itertools
import math


def generate_settings(mapping_file, output_file="settings.xml"):
    tree = ET.parse(mapping_file)
    root = tree.getroot()

    bead_types = sorted({
        bead.findtext("type").strip()
        for bead in root.findall(".//cg_bead")
        if bead.findtext("type")
    })

    bonded_entries = []
    for kind in ["bond", "angle", "dihedral"]:
        for entry in root.findall(f".//cg_bonded/{kind}"):
            name = entry.findtext("name")
            beads = entry.findtext("beads")
            if name and beads:
                bonded_entries.append((kind, name.strip()))

    def add(parent, tag, value):
        el = ET.SubElement(parent, tag)
        el.text = str(value)
        return el

    cg = ET.Element("cg")

    fm = ET.SubElement(cg, "fmatch")
    add(fm, "constrainedLS", 0)
    add(fm, "frames_per_block", 100)

    seen = set()

    for kind, name in bonded_entries:
        key = (kind, name)
        if key in seen:
            continue
        seen.add(key)

        if kind == "bond":
            stat_min, stat_max, stat_step = 0.0, 0.40, 0.001
            fm_min, fm_max, fm_step, fm_out = 0.12, 0.22, 0.001, 0.0002

        elif kind == "angle":
            stat_min, stat_max, stat_step = 0.0, math.pi, 0.01
            fm_min, fm_max, fm_step, fm_out = 1.2, 2.4, 0.05, 0.01

        elif kind == "dihedral":
            stat_min, stat_max, stat_step = -math.pi, math.pi, 0.02
            fm_min, fm_max, fm_step, fm_out = -math.pi, math.pi, 0.05, 0.01

        else:
            continue

        bonded = ET.SubElement(cg, "bonded")
        add(bonded, "name", name)
        add(bonded, "min", stat_min)
        add(bonded, "max", stat_max)
        add(bonded, "step", stat_step)

        fm_block = ET.SubElement(bonded, "fmatch")
        add(fm_block, "min", fm_min)
        add(fm_block, "max", fm_max)
        add(fm_block, "step", fm_step)
        add(fm_block, "out_step", fm_out)

    for t1, t2 in itertools.combinations_with_replacement(bead_types, 2):
        nb = ET.SubElement(cg, "non-bonded")
        add(nb, "name", f"nb_{t1}-{t2}")
        add(nb, "min", 0.0)
        add(nb, "max", 2.0)
        add(nb, "step", 0.01)
        add(nb, "type1", t1)
        add(nb, "type2", t2)

        fm_block = ET.SubElement(nb, "fmatch")
        add(fm_block, "min", 0.30)
        add(fm_block, "max", 1.20)
        add(fm_block, "step", 0.05)
        add(fm_block, "out_step", 0.01)

    ET.indent(cg, space="  ")
    ET.ElementTree(cg).write(output_file, encoding="utf-8", xml_declaration=True)

    print(f"Generated {output_file}")
    print(f"Bead types: {bead_types}")
    print(f"Bonded entries: {len(seen)}")
    print(f"Nonbonded pairs: {len(list(itertools.combinations_with_replacement(bead_types, 2)))}")


if __name__=="__main__":
    mapping_file = Path(__file__).resolve().parent.parent / "2_mapping" / "mapping.xml"
    generate_settings(mapping_file, "settings.xml")