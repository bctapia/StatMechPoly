import numpy as np
import matplotlib.pyplot as plt


def U(x, A, B):
    return -A * x**2 + B * x**4


def run_mc(
    A,
    B,
    T=1.0,
    n_steps=2_000_000,
    step_size=0.5,
    x0=None,
    seed=None,
    burn_in=10_000,
    buffer_fraction=0.25,
):
    rng = np.random.default_rng(seed)
    kB = 1.0
    beta = 1.0 / (kB * T)

    x_min = np.sqrt(A / (2 * B))

    if x0 is None:
        x = -x_min
    else:
        x = x0

    h = buffer_fraction * x_min

    def get_state(x, old_state):
        if x < -h:
            return -1
        elif x > h:
            return 1
        else:
            return old_state

    state = get_state(x, -1)
    last_jump_step = None
    waiting_times = []

    accepted = 0

    for step in range(n_steps):
        x_trial = x + rng.uniform(-step_size, step_size)

        dU = U(x_trial, A, B) - U(x, A, B)

        if dU <= 0 or rng.random() < np.exp(-beta * dU):
            x = x_trial
            accepted += 1

        if step < burn_in:
            state = get_state(x, state)
            continue

        new_state = get_state(x, state)

        if new_state != state:
            if last_jump_step is not None:
                waiting_times.append(step - last_jump_step)
            last_jump_step = step
            state = new_state

    waiting_times = np.array(waiting_times, dtype=float)

    return {
        "A": A,
        "B": B,
        "T": T,
        "curvature": 4 * A,
        "barrier": A**2 / (4 * B),
        "mean_waiting_time": np.mean(waiting_times) if len(waiting_times) > 0 else np.nan,
        "std_waiting_time": np.std(waiting_times) if len(waiting_times) > 0 else np.nan,
        "n_jumps": len(waiting_times),
        "acceptance_rate": accepted / n_steps,
        "waiting_times": waiting_times,
    }


def average_over_trajectories(
    A,
    B,
    T=1.0,
    n_traj=20,
    n_steps=2_000_000,
    step_size=0.5,
    burn_in=10_000,
):
    all_waits = []
    acceptance_rates = []

    for i in range(n_traj):
        result = run_mc(
            A=A,
            B=B,
            T=T,
            n_steps=n_steps,
            step_size=step_size,
            seed=i,
            burn_in=burn_in,
        )

        all_waits.extend(result["waiting_times"])
        acceptance_rates.append(result["acceptance_rate"])

    all_waits = np.array(all_waits, dtype=float)

    return {
        "A": A,
        "B": B,
        "T": T,
        "curvature": 4 * A,
        "barrier": A**2 / (4 * B),
        "mean_waiting_time": np.mean(all_waits) if len(all_waits) > 0 else np.nan,
        "std_waiting_time": np.std(all_waits) if len(all_waits) > 0 else np.nan,
        "n_jumps": len(all_waits),
        "mean_acceptance_rate": np.mean(acceptance_rates),
    }


if __name__ == "__main__":

    T = 1.0

    params = [
        (0.5, 0.5),
        (0.75, 0.5),
        (1.0, 0.5),
        (1.25, 0.5),
        (1.5, 0.5),
        (1.75, 0.5),
        (2.0, 0.5),
    ]

    results = []

    for A, B in params:
        result = average_over_trajectories(
            A=A,
            B=B,
            T=T,
            n_traj=20,
            n_steps=500_000,
            step_size=0.5,
            burn_in=10_000,
        )

        results.append(result)

        print(
            f"A={A:.3f}, B={B:.3f}, "
            f"k={result['curvature']:.3f}, "
            f"dU={result['barrier']:.3f}, "
            f"<tau>={result['mean_waiting_time']:.3f}, "
            f"jumps={result['n_jumps']}, "
            f"acc={result['mean_acceptance_rate']:.3f}"
        )

    curvature = np.array([r["curvature"] for r in results])
    barrier = np.array([r["barrier"] for r in results])
    tau = np.array([r["mean_waiting_time"] for r in results])

    plt.figure()
    plt.plot(curvature, tau, "o-")
    plt.xlabel("Well curvature, k = 4A")
    plt.ylabel("Average waiting time")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("mc_waiting_time_vs_curvature.png", dpi=300)

    plt.figure()
    plt.plot(barrier, tau, "o-")
    plt.xlabel("Barrier height, ΔU = A² / 4B")
    plt.ylabel("Average waiting time")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("mc_waiting_time_vs_barrier.png", dpi=300)

    plt.show()