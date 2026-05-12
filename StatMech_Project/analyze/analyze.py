from pathlib import Path
import numpy as np
from smithlab import votca


# Directories are labeled as x_map_votca
# x = [4, 5, 6] should be skipped
# x = [7, 8, 9] can compute hybrid (exclusion method) BI (bonds, angles, dihedrals)/FM (nonbond)
def analyze_aa():
    temp = 308.15
    base_dir = Path(__file__).resolve().parent.parent / "gmx"
    bi_dir = base_dir / "aa_votca" / "5_csg_stat"
    bi_files = list(bi_dir.rglob("*.dist.new"))
    bi_new_dir = base_dir.parent / "analyze" / "aa_votca" / "bi"
    bi_new_dir.mkdir(parents=True, exist_ok=True)
    for bi_file in bi_files:
        print(f"Analyzing {bi_file.stem}")
        dist_type = bi_file.stem.split("_")[0]
        if dist_type == "nb":
            dist_type = "nonbond"
        votca.compute_bi(bi_file, temp, dist_type, output_file = bi_new_dir / bi_file.name, plot=False)


def potentials():
    temp = 308.15
    base_dir = Path(__file__).resolve().parent.parent / "gmx"
    for x in np.arange(1, 10, 1):
        print(f"================ANALYZING {x}_map_votca================")
        x = int(x)
        if x in [4,5,6]:
            continue

        votca_dir = base_dir / f"{x}_map_votca"
        # x = [1, 2, 3] can compute BI or FM using all bonds, angles, dihedrals, nonbond        
        if x in [1, 2, 3]:
            bi_dir = votca_dir / "5_csg_stat"
            bi_files = list(bi_dir.rglob("*.dist.new"))
            bi_new_dir = base_dir.parent / "analyze" / f"{x}_map_votca" / "bi"
            bi_new_dir.mkdir(parents=True, exist_ok=True)
            for bi_file in bi_files:
                print(f"Analyzing {bi_file.stem}")
                dist_type = bi_file.stem.split("_")[0]
                if dist_type == "nb":
                    dist_type = "nonbond"
                votca.compute_bi(bi_file, temp, dist_type, output_file = bi_new_dir / bi_file.name, plot=False)

            fm_dir = votca_dir / "7_csg_fmatch"
            fm_files = list(fm_dir.rglob("*.force"))
            fm_new_dir = base_dir.parent / "analyze" / f"{x}_map_votca" / "fm"
            fm_new_dir.mkdir(parents=True, exist_ok=True)
            for fm_file in fm_files:
                print(f"Analyzing {fm_file.stem}")
                dist_type = fm_file.stem.split("_")[0]
                if dist_type == "nb":
                    shift_mode = "cutoff"
                elif dist_type in ["bond", "angle", "dihedral"]:
                    shift_mode = "min"
                votca.compute_fm(fm_file, shift_mode, output_file = fm_new_dir / fm_file.name, plot=False)
        
        if x in [7, 8, 9]:
            fm_dir = votca_dir / "10_csg_fmatch"
            fm_files = list(fm_dir.rglob("*.force"))
            fm_new_dir = base_dir.parent / "analyze" / f"{x}_map_votca" / "fm"
            fm_new_dir.mkdir(parents=True, exist_ok=True)
            for fm_file in fm_files:
                print(f"Analyzing {fm_file.stem}")
                dist_type = fm_file.stem.split("_")[0]
                if dist_type == "nb":
                    shift_mode = "cutoff"
                votca.compute_fm(fm_file, shift_mode, output_file = fm_new_dir / fm_file.name, plot=False)


#analyze_aa()
#potentials()



    