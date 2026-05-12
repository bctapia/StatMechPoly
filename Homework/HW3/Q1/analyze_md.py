from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_run_name(run_dir):
    m = re.search(r"N(\d+).*_T([0-9.]+)", run_dir.name)
    if m is None:
        raise ValueError(f"Could not parse N and T from {run_dir.name}")

    return int(m.group(1)), float(m.group(2))


def read_last_lammps_thermo_block(log_file):
    """
    Reads only the final thermo block from log.lammps.
    This avoids minimization and relaxation thermo blocks.
    """
    blocks = []
    header = None
    rows = []

    with open(log_file, "r") as f:
        for line in f:
            parts = line.split()

            if not parts:
                continue

            if parts[0] == "Step":
                if header is not None and rows:
                    blocks.append((header, rows))

                header = parts
                rows = []
                continue

            if header is None:
                continue

            if line.startswith("Loop time"):
                if rows:
                    blocks.append((header, rows))

                header = None
                rows = []
                continue

            if len(parts) != len(header):
                continue

            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    if header is not None and rows:
        blocks.append((header, rows))

    if len(blocks) == 0:
        raise ValueError(f"No thermo blocks found in {log_file}")

    header, rows = blocks[-1]
    return pd.DataFrame(rows, columns=header)


def summarize_md_run(run_dir, burn_fraction=0.2):
    N, T = parse_run_name(run_dir)

    log_file = run_dir / "log.lammps"
    if not log_file.exists():
        raise FileNotFoundError(f"Missing {log_file}")

    df = read_last_lammps_thermo_block(log_file)

    rg_candidates = [
        c for c in df.columns
        if "rg" in c.lower() or "gyr" in c.lower()
    ]

    if len(rg_candidates) == 0:
        raise ValueError(
            f"{log_file} does not contain an Rg-like column. "
            f"Columns are: {list(df.columns)}"
        )

    rg_col = rg_candidates[0]

    step_min = df["Step"].min()
    step_max = df["Step"].max()
    burn_cutoff = step_min + burn_fraction * (step_max - step_min)

    prod = df[df["Step"] >= burn_cutoff].copy()

    if len(prod) < 2:
        raise ValueError(f"Not enough production samples in {log_file}")

    Rg = prod[rg_col].to_numpy()
    Rg2 = Rg**2

    Rg2_mean = np.mean(Rg2)
    Rg2_std = np.std(Rg2, ddof=1)
    var_Rg2 = np.var(Rg2, ddof=1)

    return {
        "N": N,
        "T": T,
        "interaction": 1.0 / T,
        "Rg_mean": np.mean(Rg),
        "Rg_std": np.std(Rg, ddof=1),
        "Rg2_mean": Rg2_mean,
        "Rg2_std": Rg2_std,
        "var_Rg2": var_Rg2,
        "Rg2_over_N": Rg2_mean / N,
        "relative_Rg2_fluct": Rg2_std / Rg2_mean,
        "chi_R": var_Rg2 / N,
        "n_samples": len(prod),
        "rg_column": rg_col,
        "run_dir": str(run_dir),
    }


def estimate_theta_from_slope(summary):
    """
    Estimate theta as the interaction where Rg2/N is least dependent on N.
    At theta, Rg2 ~ N, so Rg2/N should be approximately independent of N.
    """
    rows = []

    for interaction, g in summary.groupby("interaction"):
        g = g.sort_values("N")

        if len(g) < 2:
            continue

        x = g["N"].to_numpy(dtype=float)
        y = g["Rg2_over_N"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)

        rows.append({
            "interaction": interaction,
            "slope": slope,
            "abs_slope": abs(slope),
            "intercept": intercept,
            "n_points": len(g),
        })

    slope_df = pd.DataFrame(rows)

    if len(slope_df) == 0:
        return np.nan, slope_df

    best = slope_df.loc[slope_df["abs_slope"].idxmin()]
    return float(best["interaction"]), slope_df


def analyze_md_results(
    lammps_dir="lammps_runs",
    plotdir="md_plots",
    burn_fraction=0.2,
):
    lammps_dir = Path(lammps_dir)
    plotdir = Path(plotdir)
    plotdir.mkdir(exist_ok=True)

    rows = []

    for run_dir in sorted(lammps_dir.glob("N*_coil_T*")):
        try:
            print(f"Reading {run_dir}")
            rows.append(summarize_md_run(run_dir, burn_fraction=burn_fraction))
        except Exception as e:
            print(f"Skipping {run_dir}: {e}")

    if len(rows) == 0:
        raise RuntimeError("No valid MD runs found.")

    summary = pd.DataFrame(rows)

    summary["interaction"] = summary["interaction"].round(6)
    summary = summary.sort_values(["N", "interaction"])
    summary.to_csv(plotdir / "md_summary.csv", index=False)

    theta_estimate, slope_df = estimate_theta_from_slope(summary)
    slope_df.to_csv(plotdir / "md_theta_slope_scan.csv", index=False)

    with open(plotdir / "md_theta_estimate.txt", "w") as f:
        f.write(f"theta_interaction_estimate = {theta_estimate}\n")
        f.write("method = minimum absolute slope of Rg2/N vs N\n")
        if np.isfinite(theta_estimate):
            f.write(f"theta_temperature_estimate = {1.0 / theta_estimate}\n")

    # ------------------------------------------------------------
    # Part 1: Rg^2 vs interaction
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["Rg2_mean"],
            linewidth=2,
            label=f"N={N}",
        )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\langle R_g^2 \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part1_MD_Rg2_vs_interaction.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Part 2: theta crossing plot
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["Rg2_over_N"],
            linewidth=2,
            label=f"N={N}",
        )

    if np.isfinite(theta_estimate):
        plt.axvline(
            theta_estimate,
            linestyle="--",
            label=rf"$\theta$ estimate = {theta_estimate:.3f}",
        )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\langle R_g^2 \rangle/N$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part2_MD_theta_crossing.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Part 3a: raw variance vs interaction
    # Shows where fluctuations peak
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["var_Rg2"],
            linewidth=2,
            label=f"N={N}",
        )

    if np.isfinite(theta_estimate):
        plt.axvline(
            theta_estimate,
            linestyle="--",
            label=rf"$\theta$ estimate = {theta_estimate:.3f}",
        )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\mathrm{Var}(R_g^2)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part3_MD_var_Rg2_vs_interaction.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Part 3b: relative fluctuations vs interaction
    # Dimensionless fluctuation size
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["relative_Rg2_fluct"],
            linewidth=2,
            label=f"N={N}",
        )

    if np.isfinite(theta_estimate):
        plt.axvline(
            theta_estimate,
            linestyle="--",
            label=rf"$\theta$ estimate = {theta_estimate:.3f}",
        )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\sigma(R_g^2)/\langle R_g^2 \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part3_MD_relative_fluctuations.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Part 3c: susceptibility-like fluctuations vs interaction
    # Tests whether fluctuations grow beyond trivial N scaling
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["chi_R"],
            linewidth=2,
            label=f"N={N}",
        )

    if np.isfinite(theta_estimate):
        plt.axvline(
            theta_estimate,
            linestyle="--",
            label=rf"$\theta$ estimate = {theta_estimate:.3f}",
        )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\chi_R = \mathrm{Var}(R_g^2)/N$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part3_MD_chi_R_vs_interaction.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Near-theta scaling
    # ------------------------------------------------------------
    if np.isfinite(theta_estimate):
        summary["theta_distance"] = np.abs(summary["interaction"] - theta_estimate)
        near_theta = summary.loc[
            summary.groupby("N")["theta_distance"].idxmin()
        ].copy()
    else:
        middle_T = np.median(summary["T"].unique())
        near_theta = summary[summary["T"] == middle_T].copy()

    near_theta = near_theta.sort_values("N")
    near_theta.to_csv(plotdir / "md_near_theta_fluctuations.csv", index=False)

    # Scaling plot: relative fluctuation near theta
    # Markers are useful here because there are only four N values.
    plt.figure()
    plt.plot(
        near_theta["N"],
        near_theta["relative_Rg2_fluct"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("N")
    plt.ylabel(r"$\sigma(R_g^2)/\langle R_g^2 \rangle$ near $\theta$")
    plt.tight_layout()
    plt.savefig(plotdir / "part3_MD_relative_fluctuation_scaling_near_theta.png", dpi=300)
    plt.close()

    # Scaling plot: chi_R near theta
    plt.figure()
    plt.plot(
        near_theta["N"],
        near_theta["chi_R"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("N")
    plt.ylabel(r"$\chi_R = \mathrm{Var}(R_g^2)/N$ near $\theta$")
    plt.tight_layout()
    plt.savefig(plotdir / "part3_MD_chi_R_scaling_near_theta.png", dpi=300)
    plt.close()

    # Scaling plot: raw variance near theta
    plt.figure()
    plt.plot(
        near_theta["N"],
        near_theta["var_Rg2"],
        marker="o",
        linewidth=2,
    )
    plt.xlabel("N")
    plt.ylabel(r"$\mathrm{Var}(R_g^2)$ near $\theta$")
    plt.tight_layout()
    plt.savefig(plotdir / "part3_MD_var_Rg2_scaling_near_theta.png", dpi=300)
    plt.close()

    # Optional log-log fit: chi_R ~ N^alpha
    if len(near_theta) >= 2 and np.all(near_theta["chi_R"] > 0):
        logN = np.log(near_theta["N"].to_numpy(dtype=float))
        logchi = np.log(near_theta["chi_R"].to_numpy(dtype=float))
        alpha, logA = np.polyfit(logN, logchi, 1)

        with open(plotdir / "md_fluctuation_scaling_fit.txt", "w") as f:
            f.write("Fit near theta: chi_R ~ N^alpha\n")
            f.write(f"alpha = {alpha}\n")
            f.write(f"A = {np.exp(logA)}\n")

        plt.figure()
        plt.plot(
            near_theta["N"],
            near_theta["chi_R"],
            marker="o",
            linewidth=2,
            label="data",
        )

        N_fit = np.linspace(near_theta["N"].min(), near_theta["N"].max(), 200)
        chi_fit = np.exp(logA) * N_fit**alpha

        plt.plot(
            N_fit,
            chi_fit,
            linestyle="--",
            linewidth=2,
            label=rf"fit: $\alpha={alpha:.3f}$",
        )

        plt.xlabel("N")
        plt.ylabel(r"$\chi_R = \mathrm{Var}(R_g^2)/N$ near $\theta$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotdir / "part3_MD_chi_R_scaling_fit_near_theta.png", dpi=300)
        plt.close()

    print()
    print(f"Wrote {plotdir / 'md_summary.csv'}")
    print(f"Theta interaction estimate: {theta_estimate}")
    print(f"Wrote near-theta fluctuations to {plotdir / 'md_near_theta_fluctuations.csv'}")

    return summary


def compare_mc_md(
    mc_summary_file="mc_plots/mc_summary.csv",
    md_summary_file="md_plots/md_summary.csv",
    plotdir="comparison_plots",
):
    plotdir = Path(plotdir)
    plotdir.mkdir(exist_ok=True)

    mc = pd.read_csv(mc_summary_file)
    md = pd.read_csv(md_summary_file)

    mc["interaction"] = mc["interaction"].round(6)
    md["interaction"] = md["interaction"].round(6)

    # Make sure MC has these columns if old summary was generated
    if "var_Rg2" not in mc.columns:
        mc["var_Rg2"] = mc["Rg2_std"] ** 2
    if "chi_R" not in mc.columns:
        mc["chi_R"] = mc["var_Rg2"] / mc["N"]

    if "var_Rg2" not in md.columns:
        md["var_Rg2"] = md["Rg2_std"] ** 2
    if "chi_R" not in md.columns:
        md["chi_R"] = md["var_Rg2"] / md["N"]

    Ns = sorted(set(mc["N"]).union(md["N"]))

    # Plot 1: Rg^2
    plt.figure()
    for N in Ns:
        if N in mc["N"].values:
            g = mc[mc["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["Rg2_mean"],
                linestyle="--",
                linewidth=2,
                label=f"N={N} MC",
            )

        if N in md["N"].values:
            g = md[md["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["Rg2_mean"],
                linestyle="-",
                linewidth=2,
                label=f"N={N} MD",
            )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\langle R_g^2 \rangle$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plotdir / "MC_vs_MD_Rg2.png", dpi=300)
    plt.close()

    # Plot 2: Rg^2 / N
    plt.figure()
    for N in Ns:
        if N in mc["N"].values:
            g = mc[mc["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["Rg2_over_N"],
                linestyle="--",
                linewidth=2,
                label=f"N={N} MC",
            )

        if N in md["N"].values:
            g = md[md["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["Rg2_over_N"],
                linestyle="-",
                linewidth=2,
                label=f"N={N} MD",
            )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\langle R_g^2 \rangle/N$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plotdir / "MC_vs_MD_theta_crossing.png", dpi=300)
    plt.close()

    # Plot 3: relative fluctuations
    plt.figure()
    for N in Ns:
        if N in mc["N"].values:
            g = mc[mc["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["relative_Rg2_fluct"],
                linestyle="--",
                linewidth=2,
                label=f"N={N} MC",
            )

        if N in md["N"].values:
            g = md[md["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["relative_Rg2_fluct"],
                linestyle="-",
                linewidth=2,
                label=f"N={N} MD",
            )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\sigma(R_g^2)/\langle R_g^2 \rangle$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plotdir / "MC_vs_MD_relative_fluctuations.png", dpi=300)
    plt.close()

    # Plot 4: raw variance
    plt.figure()
    for N in Ns:
        if N in mc["N"].values:
            g = mc[mc["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["var_Rg2"],
                linestyle="--",
                linewidth=2,
                label=f"N={N} MC",
            )

        if N in md["N"].values:
            g = md[md["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["var_Rg2"],
                linestyle="-",
                linewidth=2,
                label=f"N={N} MD",
            )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\mathrm{Var}(R_g^2)$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plotdir / "MC_vs_MD_var_Rg2.png", dpi=300)
    plt.close()

    # Plot 5: susceptibility-like fluctuations
    plt.figure()
    for N in Ns:
        if N in mc["N"].values:
            g = mc[mc["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["chi_R"],
                linestyle="--",
                linewidth=2,
                label=f"N={N} MC",
            )

        if N in md["N"].values:
            g = md[md["N"] == N].sort_values("interaction")
            plt.plot(
                g["interaction"],
                g["chi_R"],
                linestyle="-",
                linewidth=2,
                label=f"N={N} MD",
            )

    plt.xlabel(r"interaction parameter $\epsilon/k_BT$")
    plt.ylabel(r"$\chi_R = \mathrm{Var}(R_g^2)/N$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plotdir / "MC_vs_MD_chi_R.png", dpi=300)
    plt.close()

    print(f"Wrote comparison plots to {plotdir}")


def main():
    analyze_md_results(
        lammps_dir="lammps_runs",
        plotdir="md_plots",
        burn_fraction=0.2,
    )

    if Path("mc_plots/mc_summary.csv").exists():
        compare_mc_md()


if __name__ == "__main__":
    main()