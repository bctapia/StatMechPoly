def rewrite_gro_strict(gro_in, gro_out):
    with open(gro_in) as f:
        lines = f.readlines()

    title = lines[0].rstrip()
    natoms = int(lines[1].strip())
    atom_lines = lines[2:2 + natoms]
    box_line = lines[2 + natoms].split()

    out = [title + "\n", f"{natoms:5d}\n"]

    for i, line in enumerate(atom_lines, start=1):
        p = line.split()

        resid_resname = p[0]
        atomname = p[1]
        atomid = int(p[2])
        x, y, z = map(float, p[3:6])

        resid = "".join(c for c in resid_resname if c.isdigit())
        resname = resid_resname[len(resid):]

        resid = int(resid) % 100000
        atomid = atomid % 100000
        resname = resname[:5]
        atomname = atomname[:5]

        out.append(
            f"{resid:5d}"
            f"{resname:<5s}"
            f"{atomname:>5s}"
            f"{atomid:5d}"
            f"{x:8.3f}"
            f"{y:8.3f}"
            f"{z:8.3f}\n"
        )

    out.append("".join(f"{float(v):10.5f}" for v in box_line) + "\n")

    with open(gro_out, "w") as f:
        f.writelines(out)


rewrite_gro_strict("nvt_converted.gro", "nvt_converted_strict.gro")