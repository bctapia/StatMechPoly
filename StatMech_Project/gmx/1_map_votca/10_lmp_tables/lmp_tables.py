from pathlib import Path
import numpy as np
from smithlab import votca

def trim_edges(df, x_col="x", n_trim=3):
    return df.iloc[n_trim:-n_trim].reset_index(drop=True)

def run(base_dir):
    files = [f for f in base_dir.glob('*') if f.is_file()]
    for file in files:
        table_type = file.stem.split("_")[0]
        name = file.stem.split(".")[0].replace("-", "_").lower()
        if table_type == "nb":
            table_type = "pair"

        smooth_df, U_func, F_func = votca.fit_spline(
            file,
            method="pchip",
            extrapolate=False,
            plot=False,
        )

        eps = 1e-6
        zero_minimum = True

        if table_type == "angle":
            x_min = smooth_df["x"].min()
            x_max = smooth_df["x"].max()

        elif table_type == "dihedral":
            x_min = smooth_df["x"].min() + eps # avoiding LAMMPS being upset about slight numerical overshoot > 360 deg
            x_max = smooth_df["x"].max() - eps


        elif table_type == "pair":
            x_min = smooth_df["x"].min()
            x_max = 1.2
            zero_minimum = False

        elif table_type == "bond":
            x_min = smooth_df["x"].min()
            x_max = smooth_df["x"].max()

        votca.write_lammps_table(
            outfile=f"{name}.table",
            keyword= name,
            U_func=U_func,
            F_func=F_func,
            x_min=x_min,
            x_max=x_max,
            zero_minimum=zero_minimum,
            n=1000,
            table_type=table_type,
            x_unit_scale="auto",
            )
        
if __name__=="__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent.parent / "analyze" / "1_map_votca" / "bi"
    run(base_dir)