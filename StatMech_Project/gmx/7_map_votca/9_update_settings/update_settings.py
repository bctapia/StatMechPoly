import xml.etree.ElementTree as ET
from pathlib import Path


def remove_interactions_from_settings(
    settings_in,
    settings_out="settings_nonbond.xml",
    remove=("bond", "angle", "dihedral"),
):
    settings_in = Path(settings_in)
    settings_out = Path(settings_out)

    tree = ET.parse(settings_in)
    root = tree.getroot()

    remove = set(remove)

    # VOTCA bonded entries are usually:
    # <bonded>
    #   <name>bond_name</name>
    #   ...
    # </bonded>
    #
    # with interaction type often inferred from name or <type>.
    for parent in root.iter():
        for child in list(parent):
            tag = child.tag.strip()

            remove_child = False

            if tag in remove:
                remove_child = True

            elif tag == "bonded":
                name = child.findtext("name", default="").lower()
                interaction_type = child.findtext("type", default="").lower()

                text_to_check = f"{name} {interaction_type}"

                if "bond" in remove and "bond" in text_to_check:
                    remove_child = True
                if "angle" in remove and "angle" in text_to_check:
                    remove_child = True
                if "dihedral" in remove and "dihedral" in text_to_check:
                    remove_child = True

            elif tag in {"non-bonded", "nonbonded"} and "nonbond" in remove:
                remove_child = True

            if remove_child:
                parent.remove(child)

    ET.indent(tree, space="  ")
    tree.write(settings_out, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {settings_out}")
    print(f"Removed interaction types: {sorted(remove)}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    remove_interactions_from_settings(
        settings_in = base_dir / "6_tighten_params" / "settings_tight.xml",
        settings_out = "settings_tight_nonbond.xml",
        remove=("bond", "angle", "dihedral"),
    )
