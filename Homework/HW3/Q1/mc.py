import numpy as np


class PolymerMC:
    def __init__(
        self,
        N,
        T,
        bond_k=100.0,
        bond_r0=1.0,
        epsilon=1.0,
        sigma=1.0,
        rcut=2.5,
        max_disp=0.15,
        pivot_prob=0.2,
        seed=None,
    ):
        self.N = N
        self.T = T
        self.beta = 1.0 / T

        self.bond_k = bond_k
        self.bond_r0 = bond_r0
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcut = rcut
        self.rcut2 = rcut**2
        self.max_disp = max_disp
        self.pivot_prob = pivot_prob

        self.rng = np.random.default_rng(seed)

        sr6 = (self.sigma / self.rcut) ** 6
        self.lj_shift = 4.0 * self.epsilon * (sr6**2 - sr6)

        self.pos = self.make_initial_coil()

        self.accepted = 0
        self.attempted = 0
        self.local_attempted = 0
        self.local_accepted = 0
        self.pivot_attempted = 0
        self.pivot_accepted = 0

    # ---------- Initialization ----------

    def make_initial_coil(self):
        pos = np.zeros((self.N, 3), dtype=float)
        for i in range(1, self.N):
            direction = self.rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            pos[i] = pos[i - 1] + self.bond_r0 * direction
        return pos

    def set_initial_coil(self):
        self.pos = self.make_initial_coil()

    def set_initial_compact(self, scale=0.5):
        pos = np.zeros((self.N, 3), dtype=float)
        for i in range(self.N):
            pos[i] = scale * self.rng.normal(size=3)
        self.pos = pos

    # ---------- Energies ----------

    def bond_energy_pair(self, r):
        return 0.5 * self.bond_k * (r - self.bond_r0) ** 2

    def lj_energy_pair_from_r2(self, r2):
        if r2 >= self.rcut2:
            return 0.0
        inv_r2 = (self.sigma**2) / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        return 4.0 * self.epsilon * (inv_r12 - inv_r6) - self.lj_shift

    def total_energy_of_positions(self, pos):
        e = 0.0

        # Bonded
        for i in range(self.N - 1):
            rij = pos[i + 1] - pos[i]
            r = np.linalg.norm(rij)
            e += self.bond_energy_pair(r)

        # Nonbonded
        for i in range(self.N - 2):
            for j in range(i + 2, self.N):
                rij = pos[j] - pos[i]
                r2 = np.dot(rij, rij)
                e += self.lj_energy_pair_from_r2(r2)

        return e

    def total_energy(self):
        return self.total_energy_of_positions(self.pos)

    def local_energy(self, i, trial_pos=None):
        if trial_pos is None:
            ri = self.pos[i]
        else:
            ri = trial_pos

        e = 0.0

        # Bonds
        if i > 0:
            rij = ri - self.pos[i - 1]
            r = np.linalg.norm(rij)
            e += self.bond_energy_pair(r)

        if i < self.N - 1:
            rij = self.pos[i + 1] - ri
            r = np.linalg.norm(rij)
            e += self.bond_energy_pair(r)

        # Nonbonded
        for j in range(self.N):
            if j == i:
                continue
            if abs(j - i) == 1:
                continue
            rij = self.pos[j] - ri
            r2 = np.dot(rij, rij)
            e += self.lj_energy_pair_from_r2(r2)

        return e

    # ---------- Geometry ----------

    def random_rotation_matrix(self):
        """
        Uniform random 3D rotation using axis-angle.
        """
        axis = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = self.rng.uniform(0.0, 2.0 * np.pi)

        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c

        R = np.array([
            [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ]
        ])
        return R

    # ---------- MC Moves ----------

    def attempt_local_move(self):
        i = self.rng.integers(0, self.N)

        old_pos = self.pos[i].copy()
        old_e = self.local_energy(i)

        disp = self.rng.uniform(-self.max_disp, self.max_disp, size=3)
        new_pos = old_pos + disp

        new_e = self.local_energy(i, trial_pos=new_pos)
        dE = new_e - old_e

        self.attempted += 1
        self.local_attempted += 1

        if dE <= 0.0 or self.rng.random() < np.exp(-self.beta * dE):
            self.pos[i] = new_pos
            self.accepted += 1
            self.local_accepted += 1
            return True, dE

        return False, dE

    def attempt_pivot_move(self):
        """
        Choose a pivot bead and rigidly rotate either the left or right segment.
        For simplicity, use full energy difference.
        """
        if self.N < 3:
            return False, 0.0

        pivot = self.rng.integers(1, self.N - 1)
        rotate_left = self.rng.random() < 0.5

        if rotate_left:
            indices = np.arange(0, pivot)
        else:
            indices = np.arange(pivot + 1, self.N)

        if len(indices) == 0:
            return False, 0.0

        old_e = self.total_energy()
        new_pos = self.pos.copy()

        origin = self.pos[pivot].copy()
        R = self.random_rotation_matrix()

        shifted = new_pos[indices] - origin
        rotated = shifted @ R.T
        new_pos[indices] = rotated + origin

        new_e = self.total_energy_of_positions(new_pos)
        dE = new_e - old_e

        self.attempted += 1
        self.pivot_attempted += 1

        if dE <= 0.0 or self.rng.random() < np.exp(-self.beta * dE):
            self.pos = new_pos
            self.accepted += 1
            self.pivot_accepted += 1
            return True, dE

        return False, dE

    def attempt_move(self):
        if self.rng.random() < self.pivot_prob:
            return self.attempt_pivot_move()
        return self.attempt_local_move()

    def sweep(self):
        for _ in range(self.N):
            self.attempt_move()

    # ---------- Statistics ----------

    def acceptance_ratio(self):
        return self.accepted / self.attempted if self.attempted else 0.0

    def local_acceptance_ratio(self):
        return self.local_accepted / self.local_attempted if self.local_attempted else 0.0

    def pivot_acceptance_ratio(self):
        return self.pivot_accepted / self.pivot_attempted if self.pivot_attempted else 0.0

    def tune_max_disp(self, target_low=0.3, target_high=0.6, factor=1.05):
        """
        Tune only local move amplitude during equilibration.
        """
        acc = self.local_acceptance_ratio()
        if acc < target_low:
            self.max_disp /= factor
        elif acc > target_high:
            self.max_disp *= factor

    def radius_of_gyration(self):
        rcm = np.mean(self.pos, axis=0)
        dr = self.pos - rcm
        return np.sqrt(np.mean(np.sum(dr**2, axis=1)))

    def end_to_end(self):
        return np.linalg.norm(self.pos[-1] - self.pos[0])

    def reset_counters(self):
        self.accepted = 0
        self.attempted = 0
        self.local_attempted = 0
        self.local_accepted = 0
        self.pivot_attempted = 0
        self.pivot_accepted = 0

    # ---------- Run ----------

    def run(
        self,
        n_sweeps,
        equil_sweeps=0,
        sample_every=10,
        tune_during_equil=True,
        verbose=False,
    ):
        energies = []
        rgs = []
        rees = []

        # Equilibration
        for sweep in range(equil_sweeps):
            self.sweep()

            if tune_during_equil and (sweep + 1) % 50 == 0:
                self.tune_max_disp()

            if verbose and (sweep + 1) % 1000 == 0:
                print(
                    f"[equil] sweep={sweep+1} "
                    f"acc={self.acceptance_ratio():.3f} "
                    f"local_acc={self.local_acceptance_ratio():.3f} "
                    f"pivot_acc={self.pivot_acceptance_ratio():.3f} "
                    f"max_disp={self.max_disp:.4f}"
                )

        self.reset_counters()

        # Production
        for sweep in range(n_sweeps):
            self.sweep()

            if (sweep + 1) % sample_every == 0:
                energies.append(self.total_energy())
                rgs.append(self.radius_of_gyration())
                rees.append(self.end_to_end())

            if verbose and (sweep + 1) % 1000 == 0:
                print(
                    f"[prod ] sweep={sweep+1} "
                    f"acc={self.acceptance_ratio():.3f} "
                    f"local_acc={self.local_acceptance_ratio():.3f} "
                    f"pivot_acc={self.pivot_acceptance_ratio():.3f}"
                )

        E = np.array(energies)
        Rg = np.array(rgs)
        Ree = np.array(rees)

        return {
            "energy": E,
            "Rg": Rg,
            "Ree": Ree,
            "acceptance": self.acceptance_ratio(),
            "local_acceptance": self.local_acceptance_ratio(),
            "pivot_acceptance": self.pivot_acceptance_ratio(),
            "max_disp": self.max_disp,
            "final_pos": self.pos.copy(),
            "mean_E_per_N": np.mean(E) / self.N if len(E) else np.nan,
            "mean_Rg": np.mean(Rg) if len(Rg) else np.nan,
            "mean_Ree": np.mean(Ree) if len(Ree) else np.nan,
            "Cv_per_N": ((np.mean(E**2) - np.mean(E)**2) / (self.N * self.T**2)) if len(E) else np.nan,
        }


if __name__ == "__main__":
    N = 100
    T = 1.5

    sim = PolymerMC(
        N=N,
        T=T,
        bond_k=100.0,
        bond_r0=1.0,
        epsilon=1.0,
        sigma=1.0,
        rcut=2.5,
        max_disp=0.15,
        pivot_prob=0.2,
        seed=1234,
    )

    # sim.set_initial_compact(scale=0.4)
    sim.set_initial_coil()

    out = sim.run(
        n_sweeps=20000,
        equil_sweeps=5000,
        sample_every=20,
        tune_during_equil=True,
        verbose=True,
    )

    print("\nResults")
    print(f"Acceptance ratio       : {out['acceptance']:.3f}")
    print(f"Local acceptance       : {out['local_acceptance']:.3f}")
    print(f"Pivot acceptance       : {out['pivot_acceptance']:.3f}")
    print(f"Final max_disp         : {out['max_disp']:.4f}")
    print(f"<E>/N                  : {out['mean_E_per_N']:.5f}")
    print(f"<Rg>                   : {out['mean_Rg']:.5f}")
    print(f"<Ree>                  : {out['mean_Ree']:.5f}")
    print(f"Cv/N                   : {out['Cv_per_N']:.5f}")