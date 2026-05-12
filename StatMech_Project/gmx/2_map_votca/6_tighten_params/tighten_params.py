from pathlib import Path
import xml.etree.ElementTree as ET
import math


def read_dist_file(dist_file):
    """
    Read a VOTCA *.dist.new file.

    Expected format:
        x   y   i

    Returns
    -------
    xs : list[float]
    ys : list[float]
    """
    xs = []
    ys = []

    with open(dist_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue

            xs.append(x)
            ys.append(y)

    if not xs:
        raise ValueError(f"No data parsed from {dist_file}")

    return xs, ys


def infer_window_from_distribution(
    xs,
    ys,
    y_threshold_fraction=0.01,
    x_padding=0.0,
    min_width=None,
):
    """
    Infer a populated x-window from a distribution.

    Parameters
    ----------
    xs, ys : lists
        Distribution data
    y_threshold_fraction : float
        Fraction of max(y) used as threshold
    x_padding : float
        Extra padding added to both sides
    min_width : float | None
        If provided, enforce at least this width

    Returns
    -------
    x_min, x_max
    """
    ymax = max(ys)

    if ymax <= 0:
        raise ValueError("Distribution is all zeros; cannot infer window.")

    threshold = ymax * y_threshold_fraction

    populated = [x for x, y in zip(xs, ys) if y > threshold]
    if not populated:
        raise ValueError("No populated bins found above threshold.")

    x_min = min(populated) - x_padding
    x_max = max(populated) + x_padding

    if min_width is not None and (x_max - x_min) < min_width:
        center = 0.5 * (x_min + x_max)
        half = 0.5 * min_width
        x_min = center - half
        x_max = center + half

    return x_min, x_max


def get_stat_range(interaction_element):
    """
    Read the outer/statistical min/max from an interaction XML element.
    """
    min_el = interaction_element.find("min")
    max_el = interaction_element.find("max")

    if min_el is None or max_el is None:
        raise ValueError("Interaction block missing outer min/max.")

    return float(min_el.text), float(max_el.text)


def ensure_fmatch_block(interaction_element):
    fm = interaction_element.find("fmatch")
    if fm is None:
        fm = ET.SubElement(interaction_element, "fmatch")
    return fm


def set_or_create(parent, tag, value):
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    el.text = str(value)
    return el


def tighten_fmatch_ranges(
    settings_file,
    dist_dir=".",
    output_file=None,
    y_threshold_fraction=0.01,
    bond_padding=0.005,
    angle_padding=0.03,
    dihedral_padding=0.05,
    nonbond_padding=0.05,
    bond_min_width=0.02,
    angle_min_width=0.20,
    dihedral_min_width=0.40,
    nonbond_min_width=0.20,
    keep_existing_step=True,
    default_steps=None,
):
    """
    Update only the nested <fmatch> min/max values in settings.xml
    based on VOTCA *.dist.new files.

    File-name matching:
      bond.dist.new                   -> bonded name "bond"
      angle.dist.new                  -> bonded name "angle"
      lmp_001-lmp_005.dist.new        -> non-bonded name "lmp_001-lmp_005"

    Parameters
    ----------
    settings_file : str or Path
        Input settings.xml
    dist_dir : str or Path
        Directory containing *.dist.new files
    output_file : str or Path | None
        Output file path; if None, overwrite settings_file
    keep_existing_step : bool
        If True, leave existing fmatch step untouched
    default_steps : dict | None
        Optional defaults if no step exists, e.g.
        {"bonded": 0.01, "non-bonded": 0.02}
    """
    settings_file = Path(settings_file)
    dist_dir = Path(dist_dir)

    if output_file is None:
        output_file = settings_file
    else:
        output_file = Path(output_file)

    if default_steps is None:
        default_steps = {
            "bonded": 0.01,
            "non-bonded": 0.02,
        }

    tree = ET.parse(settings_file)
    root = tree.getroot()

    # Build lookup tables by interaction name
    bonded_lookup = {}
    for block in root.findall("bonded"):
        name_el = block.find("name")
        if name_el is not None and name_el.text:
            bonded_lookup[name_el.text.strip()] = block

    nonbond_lookup = {}
    for block in root.findall("non-bonded"):
        name_el = block.find("name")
        if name_el is not None and name_el.text:
            nonbond_lookup[name_el.text.strip()] = block

    updated = []
    skipped = []

    for dist_file in sorted(dist_dir.glob("*.dist.new")):
        interaction_name = dist_file.name.replace(".dist.new", "")

        if interaction_name in bonded_lookup:
            block = bonded_lookup[interaction_name]
            kind = "bonded"

            if interaction_name == "bond":
                padding = bond_padding
                min_width = bond_min_width
            elif interaction_name == "angle":
                padding = angle_padding
                min_width = angle_min_width
            elif interaction_name == "dihedral":
                padding = dihedral_padding
                min_width = dihedral_min_width
            else:
                # generic bonded fallback
                padding = angle_padding
                min_width = angle_min_width

        elif interaction_name in nonbond_lookup:
            block = nonbond_lookup[interaction_name]
            kind = "non-bonded"
            padding = nonbond_padding
            min_width = nonbond_min_width

        else:
            skipped.append((dist_file.name, "no matching interaction in settings.xml"))
            continue

        try:
            xs, ys = read_dist_file(dist_file)
            new_min, new_max = infer_window_from_distribution(
                xs,
                ys,
                y_threshold_fraction=y_threshold_fraction,
                x_padding=padding,
                min_width=min_width,
            )

            stat_min, stat_max = get_stat_range(block)
            new_min = max(new_min, stat_min)
            new_max = min(new_max, stat_max)

            if new_min >= new_max:
                skipped.append((dist_file.name, "inferred window collapsed after clamping"))
                continue

            fm = ensure_fmatch_block(block)
            set_or_create(fm, "min", f"{new_min:.8f}")
            set_or_create(fm, "max", f"{new_max:.8f}")

            if not keep_existing_step or fm.find("step") is None:
                set_or_create(fm, "step", str(default_steps[kind]))

            updated.append((interaction_name, new_min, new_max))

        except Exception as exc:
            skipped.append((dist_file.name, str(exc)))

    indent_xml(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {output_file}")
    print(f"Updated {len(updated)} interactions")
    for name, xmin, xmax in updated:
        print(f"  {name}: fmatch min={xmin:.6f}, max={xmax:.6f}")

    if skipped:
        print(f"Skipped {len(skipped)} files")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


def indent_xml(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


if __name__=="__main__":   
    base_dir = Path(__file__).resolve().parent.parent

    tighten_fmatch_ranges(
        settings_file=base_dir / "5_csg_stat" / "settings.xml",
        dist_dir=base_dir / "5_csg_stat",
        output_file="settings_tight.xml",
        y_threshold_fraction=0.01,
    )