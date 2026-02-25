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

import numpy as np


class ObstacleSAWGrowth3D:
    """
    Self-avoiding walk (SAW) grown step-by-step on Z^3 with quenched site obstacles.

    Rule (matches your prompt):
      - Prefer OPEN, unoccupied neighbors.
      - If none exist (obstacle-trapped), allow stepping onto BLOCKED but unoccupied neighbors.
      - If no unoccupied neighbors exist at all (self-trapped), restart the whole walk.

    Notes:
      - This is a *growth* ensemble (not equilibrium MCMC like pivot).
      - We track how often we were forced onto blocked sites.
    """

    def __init__(self, N, p_open=0.8, obstacle_seed=123, seed=None, max_restarts=10_000):
        self.N = int(N)
        self.p_open = float(p_open)
        if not (0.0 < self.p_open <= 1.0):
            raise ValueError("p_open must be in (0, 1].")

        self.rng = np.random.default_rng(seed)
        self.obstacle_seed = int(obstacle_seed)
        self.max_restarts = int(max_restarts)

        self.coords = None
        self.used_blocked_steps = 0  # count of steps that landed on blocked due to obstacle trap
        self.restarts_used = 0

    # ---------- obstacle field: deterministic hash (x,y,z) -> uniform[0,1) ----------
    @staticmethod
    def _mix64(x: np.uint64) -> np.uint64:
        # SplitMix64 finalizer; overflow is intentional (uint64 arithmetic)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return x

    def _u01_from_coord(self, x: int, y: int, z: int) -> float:
        ox = np.uint64((x + 2**31) & 0xFFFFFFFF)
        oy = np.uint64((y + 2**31) & 0xFFFFFFFF)
        oz = np.uint64((z + 2**31) & 0xFFFFFFFF)

        h = np.uint64(self.obstacle_seed)
        h ^= ox * np.uint64(0x9E3779B1)
        h ^= oy * np.uint64(0x85EBCA77)
        h ^= oz * np.uint64(0xC2B2AE3D)
        h = self._mix64(h)

        # top 53 bits -> float in [0,1)
        return float((h >> np.uint64(11)) * (1.0 / (1 << 53)))

    def is_open(self, pos) -> bool:
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        return self._u01_from_coord(x, y, z) < self.p_open

    # ---------- lattice helpers ----------
    @staticmethod
    def _neighbors(pos):
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        return (
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1),
        )

    def generate(self, start=(0, 0, 0)):
        """
        Grow a length-N SAW (N steps, N+1 sites) with obstacles.
        On failure (self-trap), restarts until success or max_restarts.
        """
        start = (int(start[0]), int(start[1]), int(start[2]))

        # If start is blocked, that's okay under your "allow blocked when trapped" rule,
        # but usually you want a clean start. You can enforce open start if you want.
        # For now we allow any start, but it will affect interpretation.
        for r in range(self.max_restarts):
            self.restarts_used = r
            coords = np.zeros((self.N + 1, 3), dtype=np.int32)
            coords[0] = np.array(start, dtype=np.int32)
            occ = {start}
            used_blocked = 0

            ok = True
            for i in range(1, self.N + 1):
                cur = tuple(coords[i - 1].tolist())
                nbrs = [n for n in self._neighbors(cur) if n not in occ]

                # Self-trapped: no unoccupied neighbor at all -> restart
                if not nbrs:
                    ok = False
                    break

                # Prefer open neighbors
                open_nbrs = [n for n in nbrs if self.is_open(n)]
                if open_nbrs:
                    nxt = open_nbrs[self.rng.integers(0, len(open_nbrs))]
                else:
                    # Obstacle-trapped: allow stepping onto blocked site
                    nxt = nbrs[self.rng.integers(0, len(nbrs))]
                    used_blocked += 1

                coords[i] = np.array(nxt, dtype=np.int32)
                occ.add(nxt)

            if ok:
                self.coords = coords
                self.used_blocked_steps = used_blocked
                return coords

        raise RuntimeError("Failed to grow a walk without self-trapping after many restarts.")

    def r2(self) -> float:
        if self.coords is None:
            raise RuntimeError("Call generate() first.")
        R = self.coords[-1] - self.coords[0]
        return float(np.dot(R, R))


def estimate_scaling_obstacle_growth(
    N_list,
    p_open=0.8,
    n_samples=2000,
    seed=66357,
    obstacle_seed=123,
    max_restarts=10_000,
):
    """
    Loop over multiple N (like your other scaling functions), using growth SAW w/ obstacles.

    Returns:
      N_list, R2, Rrms, frac_blocked_steps, avg_restarts
    """
    N_list = np.array(sorted(np.unique(N_list)), dtype=int)
    R2 = np.zeros_like(N_list, dtype=float)
    frac_blocked = np.zeros_like(N_list, dtype=float)
    avg_restarts = np.zeros_like(N_list, dtype=float)

    for idx, N in enumerate(N_list):
        r2_vals = np.zeros(n_samples, dtype=float)
        blocked_fracs = np.zeros(n_samples, dtype=float)
        restarts = np.zeros(n_samples, dtype=float)

        for k in range(n_samples):
            w = ObstacleSAWGrowth3D(
                N,
                p_open=p_open,
                obstacle_seed=obstacle_seed,                 # quenched disorder realization
                seed=None if seed is None else seed + 100_000 * idx + k,
                max_restarts=max_restarts,
            )
            w.generate()
            r2_vals[k] = w.r2()
            blocked_fracs[k] = w.used_blocked_steps / max(1, N)
            restarts[k] = w.restarts_used

        R2[idx] = np.mean(r2_vals)
        frac_blocked[idx] = np.mean(blocked_fracs)
        avg_restarts[idx] = np.mean(restarts)

        print(
            f"N={N:5d}  samples={n_samples:5d}  "
            f"<R^2>={R2[idx]:.2f}  "
            f"frac_blocked_steps={frac_blocked[idx]:.3f}  "
            f"avg_restarts={avg_restarts[idx]:.2f}"
        )

    return N_list, R2, np.sqrt(R2), frac_blocked, avg_restarts


def run_obstacle_growth_sweep(
    N_list,
    p_opens=(0.8, 0.5, 0.2),
    n_samples=2000,
    seed=66357,
    obstacle_seed=123,
    max_restarts=10_000,
):
    """
    Convenience wrapper: run growth-based obstacle SAW scaling for multiple p_open values.

    Returns:
      results[p_open] = (N, R2, Rrms, frac_blocked_steps, avg_restarts)
    """
    N_list = np.array(sorted(np.unique(N_list)), dtype=int)
    results = {}

    for p_open in p_opens:
        print(f"\n=== Growth obstacle SAW: p_open={p_open:.2f} (blocked={1.0-p_open:.2f}) ===")
        N, R2, Rrms, frac_blocked, avg_restarts = estimate_scaling_obstacle_growth(
            N_list,
            p_open=p_open,
            n_samples=n_samples,
            seed=seed,
            obstacle_seed=obstacle_seed,
            max_restarts=max_restarts,
        )
        results[float(p_open)] = (N, R2, Rrms, frac_blocked, avg_restarts)

    return results


import numpy as np

class ChannelSAWReptation2D:
    """
    2D SAW confined to a channel: y in [0, D-1], x unbounded.
    Uses reptation (slithering-snake) MCMC:
      - remove one end
      - add a new step at the other end
      - reject if self-intersection or wall violation
    Samples an equilibrium-ish ensemble much more reliably than growth in a strip.
    """

    def __init__(self, N, D, seed=None):
        self.N = int(N)
        self.D = int(D)
        self.rng = np.random.default_rng(seed)

        self.coords = self._init_straight()
        self.occ = set(map(tuple, self.coords.tolist()))

        self.attempted = 0
        self.accepted = 0

    def _in_channel(self, x, y):
        return 0 <= y < self.D

    @staticmethod
    def _neighbors(x, y):
        return ((x+1,y), (x-1,y), (x,y+1), (x,y-1))

    def _init_straight(self):
        """
        Initial valid walk: straight along x at mid-channel.
        """
        y0 = self.D // 2
        coords = np.zeros((self.N + 1, 2), dtype=np.int32)
        coords[:, 0] = np.arange(self.N + 1, dtype=np.int32)
        coords[:, 1] = y0
        return coords

    def step(self):
        """
        One reptation attempt.
        Randomly choose direction:
          - forward: pop tail, grow at head
          - backward: pop head, grow at tail
        """
        self.attempted += 1

        forward = bool(self.rng.integers(0, 2))

        if forward:
            # remove tail, grow at head
            removed = tuple(self.coords[0].tolist())
            head = tuple(self.coords[-1].tolist())
            body_occ = self.occ.copy()
            body_occ.remove(removed)

            # candidate new head positions
            cand = []
            hx, hy = head
            for nx, ny in self._neighbors(hx, hy):
                if not self._in_channel(nx, ny):
                    continue
                if (nx, ny) in body_occ:
                    continue
                cand.append((nx, ny))
            if not cand:
                return False

            new_head = cand[int(self.rng.integers(0, len(cand)))]
            # accept
            self.coords[:-1] = self.coords[1:]
            self.coords[-1] = np.array(new_head, dtype=np.int32)
            self.occ = body_occ
            self.occ.add(new_head)
            self.accepted += 1
            return True

        else:
            # remove head, grow at tail
            removed = tuple(self.coords[-1].tolist())
            tail = tuple(self.coords[0].tolist())
            body_occ = self.occ.copy()
            body_occ.remove(removed)

            cand = []
            tx, ty = tail
            for nx, ny in self._neighbors(tx, ty):
                if not self._in_channel(nx, ny):
                    continue
                if (nx, ny) in body_occ:
                    continue
                cand.append((nx, ny))
            if not cand:
                return False

            new_tail = cand[int(self.rng.integers(0, len(cand)))]
            # accept
            self.coords[1:] = self.coords[:-1]
            self.coords[0] = np.array(new_tail, dtype=np.int32)
            self.occ = body_occ
            self.occ.add(new_tail)
            self.accepted += 1
            return True

    def run(self, n_steps=200_000, burn_in=50_000, thin=200):
        """
        Run MCMC and return samples of R_parallel^2.
        """
        r2 = []
        for t in range(n_steps):
            self.step()
            if t >= burn_in and ((t - burn_in) % thin == 0):
                R = self.coords[-1] - self.coords[0]
                r2.append(float(R[0]**2))
        return np.array(r2, dtype=float)

    @property
    def acceptance_rate(self):
        return self.accepted / self.attempted if self.attempted else 0.0

def estimate_scaling_confined_reptation(
    N_list,
    D,
    seed=66357,
    c_relax=50,     # increase if needed
    n_blocks=3,      # how many blocks of samples after burn-in
):
    N_list = np.array(sorted(np.unique(N_list)), dtype=int)
    R2 = np.zeros_like(N_list, dtype=float)
    accs = np.zeros_like(N_list, dtype=float)

    for idx, N in enumerate(N_list):
        # scale burn-in and sampling with N^2 (reptation relaxation)
        burn_in = int(c_relax * N * N)
        run_len = int(n_blocks * c_relax * N * N)
        n_steps = burn_in + run_len
        thin = max(200, N)

        saw = ChannelSAWReptation2D(N, D, seed=seed + 17*idx)
        r2_samples = saw.run(n_steps=n_steps, burn_in=burn_in, thin=thin)

        R2[idx] = np.mean(r2_samples)
        accs[idx] = saw.acceptance_rate
        print(
            f"N={N:5d} samples={len(r2_samples):5d} acc_rate={accs[idx]:.3f} "
            f"<R_par^2>={R2[idx]:.2f} burn_in={burn_in} thin={thin}"
        )

    return N_list, R2, np.sqrt(R2), accs


# -------- fitting helper --------
def fit_loglog_slope(x, y):
    lx = np.log(x)
    ly = np.log(y)
    m, b = np.polyfit(lx, ly, 1)
    return m, b

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


def run_osaw_growth():
    ###########################################
    # Choose N values
    Ns = np.unique(np.logspace(2, 3, 8).astype(int))  # ~100..1000
    p_opens = (0.8, 0.5, 0.2)

    results = run_obstacle_growth_sweep(
        Ns,
        p_opens=p_opens,
        n_samples=2000,
        seed=66357,
        obstacle_seed=123,
        max_restarts=10_000,
    )

    ###########################################
    # Scaling fits + plot
    ###########################################

    plt.figure()

    for p_open in p_opens:
        N, R2, Rrms, frac_blocked, avg_restarts = results[p_open]

        # ---- Fit slopes ----
        m_r, b_r = fit_loglog_slope(N, Rrms)  # nu
        m_2, b_2 = fit_loglog_slope(N, R2)    # 2nu

        print("\nFITS (log-log):")
        print(f"p_open = {p_open:.2f}")
        print(f"nu from Rrms ~ N^nu:        nu  = {m_r:.4f}")
        print(f"2nu from <R^2> ~ N^(2nu): 2nu = {m_2:.4f}")
        print(f"--------------------------------------")

        # ---- Plot data + fit ----
        plt.loglog(N, R2, "o-", label=rf"$\langle R^2\rangle$ ($p_{{open}}={p_open:.2f}$)")
        plt.loglog(N, np.exp(b_2) * N**m_2, "--",
                   label=rf"fit ($p_{{open}}={p_open:.2f}$) slope {m_2:.3f}")

    plt.xlabel("N (steps)")
    plt.ylabel(r"$\langle R^2\rangle$")
    plt.title("Growth SAW with Obstacles: Scaling of End-to-End Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()
    ###########################################

def run_confined2D():
    Ns = np.unique(np.logspace(2, 3, 8).astype(int))   # ~100..1000
    Ds = (8, 16, 32)

    for D in Ds:
        print(f"\n=== Reptation confined SAW, D={D} ===")
        N, R2, Rrms, accs = estimate_scaling_confined_reptation(
            Ns, D,
            seed=66357,
            c_relax=50,     # try 200; bump to 500 if needed
            n_blocks=3       # more blocks = better stats
        )

        m_r, b_r = fit_loglog_slope(N, Rrms)
        m_2, b_2 = fit_loglog_slope(N, R2)

        print("\nFITS (log-log):")
        print(f"D = {D}")
        print(f"nu from Rrms ~ N^nu:        nu  = {m_r:.4f}")
        print(f"2nu from <R^2> ~ N^(2nu): 2nu = {m_2:.4f}")

        plt.figure()
        plt.loglog(N, R2, "o-", label=rf"$\langle R_\parallel^2\rangle$ (D={D})")
        plt.loglog(N, np.exp(b_2)*N**m_2, "--", label=rf"fit slope {m_2:.3f}")
        plt.xlabel("N")
        plt.ylabel(r"$\langle R_\parallel^2\rangle$")
        plt.title(f"2D SAW in channel (reptation), D={D}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.semilogx(N, accs, "o-")
        plt.xlabel("N")
        plt.ylabel("Acceptance rate")
        plt.title(f"Reptation acceptance vs N, D={D}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    #run_rw()
    #run_saw()
    #run_osaw_growth()
    run_confined2D()