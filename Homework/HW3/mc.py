import numpy as np
import csv
from pathlib import Path


def make_initial_chain(N, b0=1.0, mode="coil", seed=None, min_dist=0.9):
    rng = np.random.default_rng(seed)

    if mode == "coil":
        r = np.zeros((N, 3))

        for i in range(1, N):
            placed = False

            for _ in range(10000):
                step = rng.normal(size=3)
                step /= np.linalg.norm(step)
                trial = r[i - 1] + b0 * step

                if i <= 2:
                    placed = True
                    break

                d = np.linalg.norm(r[:i - 1] - trial, axis=1)

                if np.all(d > min_dist):
                    placed = True
                    break

            if not placed:
                raise RuntimeError(f"Could not place bead {i}")

            r[i] = trial

        r -= r.mean(axis=0)
        return r


def bond_energy(r, k=100.0, b0=1.0):
    b = np.linalg.norm(np.diff(r, axis=0), axis=1)
    return 0.5 * k * np.sum((b - b0) ** 2)


def lj_energy(r, epsilon=1.0, sigma=1.0, rcut=2.5):
    """
    Standard 12-6 Lennard-Jones with shift at cutoff.
    Excludes bonded neighbors (|i-j| <= 1).
    """
    N = len(r)
    e = 0.0

    sr_c = sigma / rcut
    u_shift = 4 * epsilon * (sr_c**12 - sr_c**6)

    for i in range(N - 2):
        dr = r[i + 2:] - r[i]
        d = np.linalg.norm(dr, axis=1)

        mask = d < rcut
        if np.any(mask):
            sr = sigma / d[mask]
            e += np.sum(4 * epsilon * (sr**12 - sr**6) - u_shift)

    return e


def total_energy(r, k=100.0, b0=1.0, epsilon=1.0, sigma=1.0, rcut=3.0):
    return bond_energy(r, k=k, b0=b0) + lj_energy(
        r, epsilon=epsilon, sigma=sigma, rcut=rcut
    )


def local_energy_for_bead(r, i, k=100.0, b0=1.0, epsilon=1.0, sigma=1.0, rcut=3.0):
    N = len(r)
    e = 0.0

    # Local bond contribution
    if i > 0:
        b = np.linalg.norm(r[i] - r[i - 1])
        e += 0.5 * k * (b - b0) ** 2
    if i < N - 1:
        b = np.linalg.norm(r[i + 1] - r[i])
        e += 0.5 * k * (b - b0) ** 2

    # Nonbonded contribution involving bead i
    idx = np.arange(N)
    mask = np.abs(idx - i) > 1
    dr = r[mask] - r[i]
    d = np.linalg.norm(dr, axis=1)

    sr_c = sigma / rcut
    u_shift = epsilon * (sr_c**9 - 1.5 * sr_c**3)

    inside = d < rcut
    if np.any(inside):
        sr = sigma / d[inside]
        e += np.sum(epsilon * (sr**9 - 1.5 * sr**3) - u_shift)

    return e


def radius_of_gyration(r):
    rc = r - r.mean(axis=0)
    return np.sqrt(np.mean(np.sum(rc**2, axis=1)))


def end_to_end(r):
    return np.linalg.norm(r[-1] - r[0])


def random_rotation_matrix(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)

    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c + ux*ux*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c), uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])


def mc_run(
    N,
    T,
    n_steps=2_000_000,
    sample_every=1000,
    burn_in=200_000,
    init="coil",
    seed=1234,
    k_bond=100.0,
    b0=1.0,
    epsilon=1.0,
    sigma=1.0,
    rcut=3.0,
    max_disp=0.20,
    pivot_prob=0.05,
    outdir="mc_output",
):
    rng = np.random.default_rng(seed)
    beta = 1.0 / T

    r = make_initial_chain(N, b0=b0, mode=init, seed=seed)
    E = total_energy(r, k=k_bond, b0=b0, epsilon=epsilon, sigma=sigma, rcut=rcut)

    attempted_local = accepted_local = 0
    attempted_pivot = accepted_pivot = 0

    rows = []

    for step in range(1, n_steps + 1):
        if rng.random() < pivot_prob:
            attempted_pivot += 1

            pivot = rng.integers(1, N - 1)
            rotate_right = rng.random() < 0.5

            r_trial = r.copy()
            R = random_rotation_matrix(rng)

            if rotate_right:
                segment = r[pivot + 1:] - r[pivot]
                r_trial[pivot + 1:] = r[pivot] + segment @ R.T
            else:
                segment = r[:pivot] - r[pivot]
                r_trial[:pivot] = r[pivot] + segment @ R.T

            E_trial = total_energy(
                r_trial, k=k_bond, b0=b0,
                epsilon=epsilon, sigma=sigma, rcut=rcut
            )
            dE = E_trial - E

            if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
                r = r_trial
                E = E_trial
                accepted_pivot += 1

        else:
            attempted_local += 1

            i = rng.integers(0, N)
            old_pos = r[i].copy()
            E_old_local = local_energy_for_bead(
                r, i, k=k_bond, b0=b0,
                epsilon=epsilon, sigma=sigma, rcut=rcut
            )

            disp = rng.uniform(-max_disp, max_disp, size=3)
            r[i] += disp

            E_new_local = local_energy_for_bead(
                r, i, k=k_bond, b0=b0,
                epsilon=epsilon, sigma=sigma, rcut=rcut
            )

            dE = E_new_local - E_old_local

            if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
                E += dE
                accepted_local += 1
            else:
                r[i] = old_pos

        if step % sample_every == 0:
            rows.append({
                "step": step,
                "T": T,
                "N": N,
                "E": E,
                "E_per_bead": E / N,
                "Rg": radius_of_gyration(r),
                "Ree": end_to_end(r),
                "local_acceptance": accepted_local / max(attempted_local, 1),
                "pivot_acceptance": accepted_pivot / max(attempted_pivot, 1),
                "production": step >= burn_in,
            })

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    outfile = outdir / f"mc_N{N}_T{T:.3f}_{init}.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return outfile


def main():
    chain_lengths = [50, 100, 150, 200]

    # Scan around collapse. You should refine this after seeing Rg(T).
    temperatures = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

    for N in chain_lengths:
        for T in temperatures:
            print(f"Running N={N}, T={T}")
            outfile = mc_run(
                N=N,
                T=T,
                n_steps=2_000_000,
                sample_every=1000,
                burn_in=200_000,
                init="coil",
                seed=1000 + N + int(100 * T),
                max_disp=0.20,
                pivot_prob=0.03,
                outdir="mc_output",
            )
            print(f"wrote {outfile}")


if __name__ == "__main__":
    main()