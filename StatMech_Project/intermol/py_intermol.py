from smithlab import gromacs as gro

intermol_loc = "/home/gridsan/btapia/software/InterMol/intermol/convert.py"
pair_style = "pair_style lj/charmm/coul/long 9.0 10.0"
gro.intermol(intermol_loc, "equilibrated_out_noimp.lmps", "nvt.in", pair_style, dihedral_remove=True)
gro.fix_harmonic_bonds("nvt_converted.top", "nvt_converted_fixed.top")
gro.add_fourier_dihedrals("equilibrated_out_noimp.lmps", "nvt_converted_fixed.top")
