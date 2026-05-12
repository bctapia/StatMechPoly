from pathlib import Path
from smithlab import votca


def run(bi_dir, fm_dir):


    # BI: bonded only
    for file in bi_dir.glob("*"):
        if not file.is_file():
            continue

        table_type = file.stem.split("_")[0]
        name = file.stem.split(".")[0].replace("-", "_").lower()

        if table_type == "nb":
            continue

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
            x_min = smooth_df["x"].min() + eps
            x_max = smooth_df["x"].max() - eps

        elif table_type == "bond":
            x_min = smooth_df["x"].min()
            x_max = smooth_df["x"].max()

        else:
            continue

        votca.write_lammps_table(
            outfile=f"{name}.table",
            keyword=name,
            U_func=U_func,
            F_func=F_func,
            x_min=x_min,
            x_max=x_max,
            zero_minimum=zero_minimum,
            n=1000,
            table_type=table_type,
            x_unit_scale="auto",
        )

    # FM: nonbonded only
    for file in fm_dir.glob("*"):
        if not file.is_file():
            continue

        table_type = file.stem.split("_")[0]
        name = file.stem.split(".")[0].replace("-", "_").lower()

        if table_type == "nb":
            table_type = "pair"
        else:
            continue

        smooth_df, U_func, F_func = votca.fit_spline(
            file,
            method="pchip",
            extrapolate=False,
            force_col="force",
            plot=False,
        )

        x_min = smooth_df["x"].min()
        x_max = 1.2
        zero_minimum = False

        votca.write_lammps_table(
            outfile=f"{name}.table",
            keyword=name,
            U_func=U_func,
            F_func=F_func,
            x_min=x_min,
            x_max=x_max,
            zero_minimum=zero_minimum,
            n=1000,
            table_type=table_type,
            x_unit_scale="auto",
        )


if __name__ == "__main__":
    bi_dir = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "analyze"
        / "3_map_votca"
        / "bi"
    )

    fm_dir = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "analyze"
        / "9_map_votca"
        / "fm"
    )

    run(bi_dir, fm_dir)