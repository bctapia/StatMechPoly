import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
        })

    slope_df = pd.DataFrame(rows)

    if len(slope_df) == 0:
        return np.nan, slope_df

    best = slope_df.loc[slope_df["abs_slope"].idxmin()]
    return float(best["interaction"]), slope_df


def analyze_mc_results(outdir="mc_output", plotdir="mc_plots"):
    outdir = Path(outdir)
    plotdir = Path(plotdir)
    plotdir.mkdir(exist_ok=True)

    files = sorted(outdir.glob("mc_N*_T*_coil.csv"))

    if len(files) == 0:
        raise FileNotFoundError(f"No MC CSV files found in {outdir}")

    frames = []

    for file in files:
        df = pd.read_csv(file)
        df = df[df["production"] == True].copy()
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["interaction"] = data["interaction"].round(6)

    summary = (
        data.groupby(["N", "T", "interaction"])
        .agg(
            Rg2_mean=("Rg2", "mean"),
            Rg2_std=("Rg2", "std"),
            Rg_mean=("Rg", "mean"),
            Rg_std=("Rg", "std"),
            E_mean=("E", "mean"),
            E_std=("E", "std"),
            local_acc=("local_acceptance", "mean"),
            pivot_acc=("pivot_acceptance", "mean"),
        )
        .reset_index()
    )

    summary["Rg2_over_N"] = summary["Rg2_mean"] / summary["N"]

    # Fluctuation measures for part (3)
    summary["var_Rg2"] = summary["Rg2_std"] ** 2
    summary["relative_Rg2_fluct"] = summary["Rg2_std"] / summary["Rg2_mean"]

    # Susceptibility-like size fluctuation measure
    # This is the main metric for testing whether fluctuations scale with N.
    summary["chi_R"] = summary["var_Rg2"] / summary["N"]

    theta_estimate, slope_df = estimate_theta_from_slope(summary)

    slope_df.to_csv(plotdir / "theta_slope_scan.csv", index=False)
    summary.to_csv(plotdir / "mc_summary.csv", index=False)

    with open(plotdir / "theta_estimate.txt", "w") as f:
        f.write(f"theta_interaction_estimate = {theta_estimate}\n")
        if np.isfinite(theta_estimate):
            f.write(f"theta_temperature_estimate = {1.0 / theta_estimate}\n")

    # ------------------------------------------------------------
    # (1) Rg^2 vs interaction parameter
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["Rg2_mean"],
            #marker="o",
            label=f"N={N}",
        )

    plt.xlabel(r"interaction parameter $\epsilon / k_B T$")
    plt.ylabel(r"$\langle R_g^2 \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part1_Rg2_vs_interaction.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # (2) theta crossing plot: Rg^2 / N
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["Rg2_over_N"],
            #marker="o",
            label=f"N={N}",
        )

    if np.isfinite(theta_estimate):
        plt.axvline(
            theta_estimate,
            linestyle="--",
            label=rf"$\theta$ estimate = {theta_estimate:.3f}",
        )

    plt.xlabel(r"interaction parameter $\epsilon / k_B T$")
    plt.ylabel(r"$\langle R_g^2 \rangle / N$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part2_theta_crossing.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # (3a) relative fluctuations vs interaction parameter
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["relative_Rg2_fluct"],
            #marker="o",
            label=f"N={N}",
        )

    if np.isfinite(theta_estimate):
        plt.axvline(
            theta_estimate,
            linestyle="--",
            label=rf"$\theta$ estimate = {theta_estimate:.3f}",
        )

    plt.xlabel(r"interaction parameter $\epsilon / k_B T$")
    plt.ylabel(r"$\sigma(R_g^2) / \langle R_g^2 \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part3_relative_fluct_vs_interaction.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # (3b) susceptibility-like fluctuations vs interaction parameter
    # ------------------------------------------------------------
    plt.figure()
    for N, g in summary.groupby("N"):
        g = g.sort_values("interaction")
        plt.plot(
            g["interaction"],
            g["chi_R"],
            #marker="o",
            label=f"N={N}",
        )

    if np.isfinite(theta_estimate):
        plt.axvline(
            theta_estimate,
            linestyle="--",
            label=rf"$\theta$ estimate = {theta_estimate:.3f}",
        )

    plt.xlabel(r"interaction parameter $\epsilon / k_B T$")
    plt.ylabel(r"$\chi_R = \mathrm{Var}(R_g^2)/N$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotdir / "part3_chi_R_vs_interaction.png", dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # (3c) fluctuation scaling near theta
    # ------------------------------------------------------------
    if np.isfinite(theta_estimate):
        summary["theta_distance"] = np.abs(summary["interaction"] - theta_estimate)
        near_theta = summary.loc[summary.groupby("N")["theta_distance"].idxmin()].copy()
    else:
        middle_T = np.median(summary["T"].unique())
        near_theta = summary[summary["T"] == middle_T].copy()

    near_theta = near_theta.sort_values("N")
    near_theta.to_csv(plotdir / "near_theta_fluctuations.csv", index=False)

    plt.figure()
    plt.plot(
        near_theta["N"],
        near_theta["relative_Rg2_fluct"],
        #marker="o",
    )
    plt.xlabel("N")
    plt.ylabel(r"$\sigma(R_g^2) / \langle R_g^2 \rangle$ near $\theta$")
    plt.tight_layout()
    plt.savefig(plotdir / "part3_relative_fluct_scaling_near_theta.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(
        near_theta["N"],
        near_theta["chi_R"],
        #marker="o",
    )
    plt.xlabel("N")
    plt.ylabel(r"$\chi_R = \mathrm{Var}(R_g^2)/N$ near $\theta$")
    plt.tight_layout()
    plt.savefig(plotdir / "part3_chi_R_scaling_near_theta.png", dpi=300)
    plt.close()

    # Optional log-log fit for critical scaling
    # chi_R ~ N^alpha
    if len(near_theta) >= 2 and np.all(near_theta["chi_R"] > 0):
        logN = np.log(near_theta["N"].to_numpy(dtype=float))
        logchi = np.log(near_theta["chi_R"].to_numpy(dtype=float))
        alpha, logA = np.polyfit(logN, logchi, 1)

        with open(plotdir / "fluctuation_scaling_fit.txt", "w") as f:
            f.write("Fit near theta: chi_R ~ N^alpha\n")
            f.write(f"alpha = {alpha}\n")
            f.write(f"A = {np.exp(logA)}\n")

        plt.figure()
        plt.plot(near_theta["N"], near_theta["chi_R"], marker="o", label="data")

        N_fit = np.linspace(near_theta["N"].min(), near_theta["N"].max(), 200)
        chi_fit = np.exp(logA) * N_fit ** alpha

        plt.plot(N_fit, chi_fit, linestyle="--", label=rf"fit: $\alpha={alpha:.3f}$")
        plt.xlabel("N")
        plt.ylabel(r"$\chi_R = \mathrm{Var}(R_g^2)/N$ near $\theta$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotdir / "part3_chi_R_scaling_fit_near_theta.png", dpi=300)
        plt.close()

    print(f"Wrote summary to {plotdir / 'mc_summary.csv'}")
    print(f"Theta interaction estimate: {theta_estimate}")
    print(f"Wrote near-theta fluctuations to {plotdir / 'near_theta_fluctuations.csv'}")

    return summary


if __name__ == "__main__":
    analyze_mc_results(
        outdir="mc_output",
        plotdir="mc_plots",
    )