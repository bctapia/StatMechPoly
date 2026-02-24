import numpy as np
import matplotlib.pyplot as plt


class RandomWalk():

    def __init__(self, N, seed=None):
        self.N = N
        self.rng = np.random.default_rng(seed)

        self.coords = None

    def run(self):
        """
        Generate a random walk of length N.
        """

        moves = np.array([
            [ 1, 0, 0],
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 0,-1, 0],
            [ 0, 0, 1],
            [ 0, 0,-1],
        ], dtype=np.int32)

        directions = self.rng.integers(0, 6, size=self.N)
        step_vectors = moves[directions]

        coords = np.zeros((self.N + 1, 3), dtype=np.int32)
        coords[1:] = np.cumsum(step_vectors, axis=0)

        self.coords = coords

        R = self.coords[-1] - self.coords[0]
        return float(np.dot(R, R))


def estimate_scaling_random_walk(
    N_list,
    n_samples=5000,
    seed=66357
):
    """
    For each N, generate many independent random walks
    and estimate <R^2>.
    """
    N_list = np.array(sorted(np.unique(N_list)), dtype=int)

    R2 = np.zeros_like(N_list, dtype=float)

    for idx, N in enumerate(N_list):
        rng_seed = None if seed is None else seed + 31 * idx

        r2_vals = np.zeros(n_samples, dtype=float)

        for k in range(n_samples):
            rw = RandomWalk(N, seed=None if rng_seed is None else rng_seed + k)
            r2_vals[k] = rw.run()

        R2[idx] = np.mean(r2_vals)

        print(f"N={N:5d}  samples={n_samples:5d}  <R^2>={R2[idx]:.2f}")

    return N_list, R2, np.sqrt(R2)


#################################################################
# ----------------------------
# Lattice symmetries: 3D signed permutation matrices
# (all 48 rotations/reflections that map Z^3 -> Z^3)
# If you want only proper rotations (det=+1), you can filter, but 48 is fine.
# ----------------------------
def signed_permutation_matrices_3d():
    """Generate all 48 signed permutation matrices in 3D
    Row i has a single nonzero entry which is either +1 or -1, and each column has exactly one nonzero entry.
    These represent all symmetries of the cubic lattice (rotations and reflections).
    """
    mats = []
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]

    for p in perms:
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    M = np.zeros((3, 3), dtype=np.int16)
                    M[0, p[0]] = sx
                    M[1, p[1]] = sy
                    M[2, p[2]] = sz
                    mats.append(M)
    # remove duplicates (shouldn’t be any, but safe)
    #uniq = []
    #seen = set()
    #for M in mats:
    #    key = tuple(M.ravel().tolist())
    #    if key not in seen:
    #        seen.add(key)
    #        uniq.append(M)
    return mats


SYMMETRIES = signed_permutation_matrices_3d()


class PivotSAW:
    def __init__(self, N, seed=None):
        self.N = N
        self.rng = np.random.default_rng(seed)

        self.coords = self.initial_straight_walk(N)
        self.occupied = self.build_occupied_set(self.coords)

        self.accepted = 0
        self.attempted = 0

    @staticmethod
    def initial_straight_walk(N):
        """Generate the initial straight walk along the x-axis:
            (0,0,0) -> (1,0,0) -> ... -> (N,0,0)
        """

        coords = np.zeros((N + 1, 3), dtype=np.int32)
        coords[:, 0] = np.arange(N + 1, dtype=np.int32)
        return coords

    @staticmethod
    def build_occupied_set(coords):
        # Python set of tuples for O(1) membership tests
        return set(map(tuple, coords.tolist()))

    def pivot_move(self):
        """
        Attempt one pivot move.
        Choose pivot site i in [0..N], choose a symmetry M, apply to one side (tail or head).
        Accept if self-avoidance preserved.

        Returns (accepted: bool, new_coords, new_occupied).
        """
        N = self.N
        coords = self.coords

        i = int(self.rng.integers(0, N + 1))  # Randomly choose pivot site i in [0...N]
        M = SYMMETRIES[int(self.rng.integers(0, len(SYMMETRIES)))]  # Randomly choose a symmetry

        # Choose which side to transform: tail (i+1..N) or head (0..i-1)
        transform_tail = bool(self.rng.integers(0, 2))

        pivot = coords[i].copy()

        if transform_tail:
            fixed_indices = np.arange(0, i + 1)  # keep 0..i
            move_indices = np.arange(i + 1, N + 1)  # transform i+1..N
        else:
            fixed_indices = np.arange(i, N + 1)  # keep i..N
            move_indices = np.arange(0, i)  # transform 0..i-1

        # Don't attempt move if no sites to move
        if move_indices.size == 0:
            return False

        # Build a set of fixed occupied sites (remove move part from occupied)
        fixed_occ = set(tuple(coords[j]) for j in fixed_indices)

        # Compute transformed coordinates for the moving part:
        # r' = pivot + M @ (r - pivot)
        rel = coords[move_indices] - pivot  # (m,3)
        rel_new = rel @ M.T  # apply linear transform
        new_part = pivot + rel_new  # (m,3)

        # Check for self-intersections:
        # 1) within new_part itself (duplicates)
        # 2) between new_part and fixed_occ
        new_tuples = [tuple(x) for x in new_part.tolist()]
        if len(set(new_tuples)) != len(new_tuples):
            return False

        for t in new_tuples:
            if t in fixed_occ:
                return False

        # Accept: assemble new coords
        self.coords[move_indices] = new_part
        self.occupied = fixed_occ.union(new_tuples)

        return True

    def run(self, n_steps, burn_in, thin):
        """
        Run pivot MCMC for a SAW of length N.
        Returns samples of R^2 and optional acceptance stats.
        """
        r2_samples = []

        for step in range(n_steps):
            self.attempted += 1
            if self.pivot_move():
                self.accepted += 1

            # sample after burn-in, with thinning
            if step >= burn_in and ((step - burn_in) % thin == 0):
                R = self.coords[-1] - self.coords[0]
                r2_samples.append(float(np.dot(R, R)))

        return np.array(r2_samples)

    @property
    def acceptance_rate(self):
        if self.attempted == 0:
            return 0.0
        return self.accepted / self.attempted


def estimate_scaling_pivot(N_list, n_steps=600_000, burn_in=150_000, thin=500, seed=66357):
    """
    For each N in N_list, run a pivot chain and estimate <R^2>.
    """
    N_list = np.array(sorted(np.unique(N_list)), dtype=int)
    R2 = np.zeros_like(N_list, dtype=float)
    accs = np.zeros_like(N_list, dtype=float)

    for idx, N in enumerate(N_list):
        run_seed = None if seed is None else seed + 12 * idx
        saw = PivotSAW(N, seed=run_seed)

        r2_samples = saw.run(n_steps=n_steps, burn_in=burn_in, thin=thin)

        R2[idx] = np.mean(r2_samples)
        accs[idx] = saw.acceptance_rate
        print(f"N={N:5d}  samples={len(r2_samples):5d}  acc_rate={accs[idx]:.3f}  <R^2>={R2[idx]:.2f}")

    return N_list, R2, np.sqrt(R2), accs


def fit_loglog_slope(x, y):
    lx = np.log(x)
    ly = np.log(y)
    m, b = np.polyfit(lx, ly, 1)
    return m, b
#################################################################


class ObstacleField3D:
    """
    Quenched random obstacles on Z^3:
      open with probability p_open, blocked otherwise.

    Implemented as a deterministic hash of (x,y,z, seed) -> uniform[0,1).
    No memory growth with explored volume.
    """
    def __init__(self, p_open, seed):
        if not (0.0 < p_open <= 1.0):
            raise ValueError("p_open must be in (0,1])")
        self.p_open = float(p_open)
        self.seed = int(seed)

    @staticmethod
    def mix64(x):
        # SplitMix64 finalizer (good bit mixing)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return x

    def u01_from_coord(self, x, y, z):
        # Pack signed ints into uint64 space in a stable way
        # (offset by 2^31 to keep non-negative within 32-bit)
        ox = np.uint64((x + 2**31) & 0xFFFFFFFF)
        oy = np.uint64((y + 2**31) & 0xFFFFFFFF)
        oz = np.uint64((z + 2**31) & 0xFFFFFFFF)

        h = np.uint64(self.seed) ^ (ox * np.uint64(0x9E3779B1)) ^ (oy * np.uint64(0x85EBCA77)) ^ (oz * np.uint64(0xC2B2AE3D))
        h = self.mix64(h)

        # Convert top 53 bits to float in [0,1)
        return float((h >> np.uint64(11)) * (1.0 / (1 << 53)))

    def is_open(self, pos):
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        return self.u01_from_coord(x, y, z) < self.p_open

    def is_blocked(self, pos):
        return not self.is_open(pos)


class ObstaclePivotSAW(PivotSAW):
    """
    Pivot SAW on a lattice with quenched obstacles.

    Rule:
      - Normally, monomers must lie on OPEN sites.
      - If we appear trapped (too many consecutive rejections), allow landing on blocked sites
        for one accepted move (still keep self-avoidance).
    """

    def __init__(self, N, p_open=0.8, obstacle_seed=123, seed=None, trap_attempts=2000):
        super().__init__(N, seed=seed)
        self.field = ObstacleField3D(p_open=p_open, seed=obstacle_seed)
        self.trap_attempts = int(trap_attempts)
        self.consec_rejects = 0
        self.escape_accepts = 0

        # Ensure initial coords are on open sites; if not, rebuild.
        if not self.all_open(self.coords):
            print("Straight initial walk hits blocked sites; regrowing open-only SAW.")
            self.build_initial_walk_with_obstacles()

    def all_open(self, coords):
        return all(self.field.is_open(p) for p in coords)

    def build_initial_walk_with_obstacles(self, max_restarts=10_000):
        """
        Build an initial SAW of length N using ONLY OPEN sites.
        If stuck, restart the whole growth process.
        """
        N = self.N
        rng = self.rng

        moves = np.array([
            [ 1, 0, 0],
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 0,-1, 0],
            [ 0, 0, 1],
            [ 0, 0,-1],
        ], dtype=np.int32)

        # Ensure starting site is open; if not, search nearby for an open start
        start = np.array([0, 0, 0], dtype=np.int32)
        if self.field.is_blocked(start):
            found = False
            for r in range(1, 50):
                # sample a few random points on the cube shell
                for _ in range(2000):
                    trial = rng.integers(-r, r + 1, size=3, dtype=np.int32)
                    if np.max(np.abs(trial)) != r:
                        continue
                    if self.field.is_open(trial):
                        start = trial
                        found = True
                        break
                if found:
                    break
            if not found:
                raise RuntimeError("Could not find an open starting site near the origin.")

        for _restart in range(max_restarts):
            coords = np.zeros((N + 1, 3), dtype=np.int32)
            coords[0] = start
            occ = {tuple(start.tolist())}

            stuck = False
            for i in range(1, N + 1):
                cur = coords[i - 1]
                order = rng.permutation(6)

                placed = False
                for j in order:
                    nxt = cur + moves[j]
                    t = tuple(nxt.tolist())
                    if t in occ:
                        continue
                    if self.field.is_blocked(nxt):
                        continue
                    coords[i] = nxt
                    occ.add(t)
                    placed = True
                    break

                if not placed:
                    stuck = True
                    break

            if not stuck:
                self.coords = coords
                self.occupied = set(map(tuple, coords.tolist()))
                return

        raise RuntimeError("Failed to grow an open-only initial walk after many restarts.")

    def pivot_move(self):
        """
        Like Saw.pivot_move(), but rejects moves landing on blocked sites,
        unless we are in "trap escape" mode for this proposal.
        """
        N = self.N
        coords = self.coords

        i = int(self.rng.integers(0, N + 1))
        M = SYMMETRIES[int(self.rng.integers(0, len(SYMMETRIES)))]
        transform_tail = bool(self.rng.integers(0, 2))
        pivot = coords[i].copy()

        if transform_tail:
            fixed_indices = np.arange(0, i + 1)
            move_indices = np.arange(i + 1, N + 1)
        else:
            fixed_indices = np.arange(i, N + 1)
            move_indices = np.arange(0, i)

        if move_indices.size == 0:
            self.consec_rejects += 1
            return False

        fixed_occ = set(tuple(coords[j]) for j in fixed_indices)

        rel = coords[move_indices] - pivot
        new_part = pivot + (rel @ M.T)

        new_tuples = [tuple(x) for x in new_part.tolist()]
        if len(set(new_tuples)) != len(new_tuples):
            self.consec_rejects += 1
            return False
        if any(t in fixed_occ for t in new_tuples):
            self.consec_rejects += 1
            return False

        # Obstacle rule: enforce open sites unless "trapped"
        trapped = (self.consec_rejects >= self.trap_attempts)
        if trapped:
            self.escape_accepts += 1
        if not trapped:
            if any(self.field.is_blocked(p) for p in new_part):
                self.consec_rejects += 1
                return False
        # If trapped: we allow blocked sites for this accepted move.

        # accept
        self.coords[move_indices] = new_part
        self.occupied = fixed_occ.union(new_tuples)

        self.consec_rejects = 0
        return True


def run_obstacle_sweep(N_list, p_opens=(0.8, 0.5, 0.2),
                       n_steps=250_000, burn_in=50_000, thin=200,
                       seed=66357, obstacle_seed=123, trap_attempts=2000):
    results = {}
    N_list = np.array(sorted(np.unique(N_list)), dtype=int)

    for p_open in p_opens:
        R2 = np.zeros_like(N_list, dtype=float)
        accs = np.zeros_like(N_list, dtype=float)

        print(f"\n=== p_open = {p_open:.2f} (blocked = {1-p_open:.2f}) ===")
        for idx, N in enumerate(N_list):
            saw = ObstaclePivotSAW(
                N,
                p_open=p_open,
                obstacle_seed=obstacle_seed,   # SAME environment across N if you want
                seed=seed + 17*idx,
                trap_attempts=trap_attempts
            )
            r2_samples = saw.run(n_steps=n_steps, burn_in=burn_in, thin=thin)
            R2[idx] = np.mean(r2_samples)
            accs[idx] = saw.acceptance_rate
            print(f"N={N:5d}  samples={len(r2_samples):5d}  acc_rate={accs[idx]:.3f}  <R^2>={R2[idx]:.2f}")

        results[p_open] = (N_list.copy(), R2, np.sqrt(R2), accs)

    return results


#########################################
#       Functions to call in main       #
#########################################

def run_rw():
    Ns = np.unique(np.logspace(2.5, 4.0, 10).astype(int))

    N, R2, Rrms = estimate_scaling_random_walk(
        Ns,
        n_samples=5000,
        seed=66357
    )

    # Fit slopes
    m_r, b_r = fit_loglog_slope(N, Rrms)  # slope = nu
    m_2, b_2 = fit_loglog_slope(N, R2)    # slope = 2nu

    print("\nFITS (log-log):")
    print(f"nu from Rrms ~ N^nu:        nu = {m_r:.4f}  (expected 0.5)")
    print(f"2nu from <R^2> ~ N^(2nu): 2nu = {m_2:.4f}  (expected 1.0)")

    # Plot scaling
    plt.figure()
    plt.loglog(N, R2, "o-", label=r"$\langle R^2\rangle$ (RW)")
    plt.loglog(N, np.exp(b_2) * N**m_2, "--", label=rf"fit slope {m_2:.3f}")
    plt.xlabel("N (steps)")
    plt.ylabel(r"$\langle R^2\rangle$")
    plt.title("3D Random Walk: Scaling of End-to-End Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_saw():
    ###########################################
    # Choose N values (keep modest for first run)
    Ns = np.unique(np.logspace(2.5, 3, 10).astype(int))  # ~50 to 1000

    # Pivot parameters:
    # - Increase n_steps for better statistics; acceptance drops slowly with N.
    N, R2, Rrms, accs = estimate_scaling_pivot(Ns, n_steps=250_000, burn_in=50_000, thin=200, seed=66357)

    # Fit exponents
    m_r, b_r = fit_loglog_slope(N, Rrms)  # slope is nu
    m_2, b_2 = fit_loglog_slope(N, R2)    # slope is 2*nu

    print("\nFITS (log-log):")
    print(f"nu from Rrms ~ N^nu:        nu = {m_r:.4f}  (expected ~0.588)")
    print(f"2nu from <R^2> ~ N^(2nu): 2nu = {m_2:.4f}  (expected ~1.176)")

    # Plot scaling
    plt.figure()
    plt.loglog(N, R2, "o-", label=r"$\langle R^2\rangle$ (pivot SAW)")
    plt.loglog(N, np.exp(b_2) * N**m_2, "--", label=rf"fit slope {m_2:.3f}")
    plt.xlabel("N (steps)")
    plt.ylabel(r"$\langle R^2\rangle$")
    plt.title("3D Self-Avoiding Walk (Pivot Algorithm): Scaling of End-to-End Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: show acceptance vs N
    plt.figure()
    plt.semilogx(N, accs, "o-")
    plt.xlabel("N")
    plt.ylabel("Pivot acceptance rate")
    plt.title("Pivot acceptance rate vs N")
    plt.tight_layout()
    plt.show()
    ###########################################


def run_osaw():
    ###########################################
    # Choose N values (keep modest for first run)
    Ns = np.unique(np.logspace(2.5, 4.0, 10).astype(int))  # ~316 to 10000

    # Obstacle parameters: fraction of OPEN sites
    p_opens = (0.8, 0.5, 0.2)

    # Run obstacle sweeps (returns dict: p_open -> (N, R2, Rrms, accs))
    results = run_obstacle_sweep(Ns, p_opens=p_opens, n_steps=250_000, burn_in=50_000, thin=200, seed=66357, obstacle_seed=123, trap_attempts=2000,)

    # --------
    # Fits + scaling plot (R2)
    # --------
    plt.figure()

    for p_open in p_opens:
        N, R2, Rrms, accs = results[p_open]

        # Fit exponents
        m_r, b_r = fit_loglog_slope(N, Rrms)  # slope is nu
        m_2, b_2 = fit_loglog_slope(N, R2)    # slope is 2*nu

        print(f"\nFITS (log-log) for p_open={p_open:.2f}:")
        print(f"nu from Rrms ~ N^nu:        nu = {m_r:.4f}")
        print(f"2nu from <R^2> ~ N^(2nu): 2nu = {m_2:.4f}")

        # Plot scaling + fit for each p_open
        plt.loglog(N, R2, "o-", label=rf"$\langle R^2\rangle$ ($p_{{open}}={p_open:.2f}$)")
        plt.loglog(N, np.exp(b_2) * N**m_2, "--", label=rf"fit ($p_{{open}}={p_open:.2f}$) slope {m_2:.3f}")

    plt.xlabel("N (steps)")
    plt.ylabel(r"$\langle R^2\rangle$")
    plt.title("3D SAW with Obstacles (Pivot): Scaling of End-to-End Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------
    # Acceptance plot
    # --------
    plt.figure()

    for p_open in p_opens:
        N, R2, Rrms, accs = results[p_open]
        plt.semilogx(N, accs, "o-", label=rf"$p_{{open}}={p_open:.2f}$")

    plt.xlabel("N")
    plt.ylabel("Pivot acceptance rate")
    plt.title("Pivot acceptance rate vs N (with obstacles)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    ###########################################


if __name__ == "__main__":
    #run_rw()
    #run_saw()
    run_osaw()
