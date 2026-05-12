from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def merge_md():
    files = sorted(Path("md_data").glob("md_results_*.csv"))

    rows = []
    header = None

    for file in files:
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            rows.extend(reader)

    with open("md_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(files)} files into md_results.csv")

def load(filename):
    return np.genfromtxt(filename, delimiter=",", names=True)


def plot(data, label):
    curvature = data["curvature"]
    barrier = data["barrier"]
    tau = data["mean_waiting_time"]

    mask = np.isfinite(tau) & (tau > 0)

    curvature = curvature[mask]
    barrier = barrier[mask]
    tau = tau[mask]

    # waiting time vs curvature
    plt.figure(1)
    plt.semilogy(curvature, tau, "o-", label=label)

    # waiting time vs barrier
    plt.figure(2)
    plt.semilogy(barrier, tau, "o-", label=label)

    plt.figure(3)
    plt.semilogy(curvature**2, tau, "o-", label=label)


if __name__ == "__main__":

    merge_md()
    md = load("md_results.csv")
    mc = load("mc_results.csv")

    plot(md, "MD")
    plot(mc, "MC")

    plt.figure(1)
    plt.xlabel("Curvature (k = 4A)")
    plt.ylabel("Waiting time")
    plt.legend()
    plt.savefig("curv_vs_logtau.png", dpi=300)

    plt.figure(2)
    plt.xlabel("Barrier (ΔU)")
    plt.ylabel("Waiting time")
    plt.legend()
    plt.savefig("barrier_vs_logtau.png", dpi=300)

    plt.figure(3)
    plt.xlabel("Curvature²")
    plt.ylabel("Waiting time")
    plt.legend()
    plt.savefig("curv_squared_vs_logtau.png", dpi=300)

    plt.tight_layout()
    plt.show()
