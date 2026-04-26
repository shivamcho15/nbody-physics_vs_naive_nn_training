"""
The system class holds the state, including positions velocities and masses
It also has two integrators (euler and leapfrog)
It also has energy/momentum diagnostics to make sure energy and momentum are conserved

G=1
"""

import numpy as np

class System:
    def __init__(self, positions, velocities, masses, G=1.0, softening=1e-3):
        # positions: (N, 2) array, one (x, y) row per body
        # velocities: (N, 2) array, one (vx, vy) row per body
        # masses: (N,1) array
        self.pos = np.array(positions, dtype=np.float64)
        self.vel = np.array(velocities, dtype=np.float64)
        self.mass = np.array(masses, dtype=np.float64)
        self.G = G
        # softening keeps forces finite when two bodies get really close.
        # Without it, a near-collision sends the integrator off to infinity.
        self.softening = softening
        self.N = len(masses)

        # sanity checks (these saved me a lot of debugging)
        assert self.pos.shape == (self.N, 2), f"bad positions shape: {self.pos.shape}"
        assert self.vel.shape == (self.N, 2), f"bad velocities shape: {self.vel.shape}"

    def compute_accelerations(self):
        """
        Returns the (N, 2) acceleration on each body due to gravity from all
        the others.

        For body i, the acceleration is:
            a_i = G * sum_{j != i} m_j * (r_j - r_i) / |r_j - r_i|^3
        """
        # Vectorized version using broadcasting.
        # diff[i, j] is the vector from body i to body j.
        # pos[:, None, :] has shape (N, 1, 2)
        # pos[None, :, :] has shape (1, N, 2)
        # Subtracting gives (N, N, 2).
        diff = self.pos[None, :, :] - self.pos[:, None, :]  # (N, N, 2)

        # squared distance with softening, shape (N, N)
        dist_sq = np.sum(diff ** 2, axis=-1) + self.softening ** 2

        # 1 / r^3 for the inverse-square law (the extra r turns the unit
        # vector r_hat into the actual displacement)
        inv_dist_cube = dist_sq ** (-1.5)

        # zero out diagonal so a body doesn't pull on itself
        np.fill_diagonal(inv_dist_cube, 0.0)

        # broadcast masses over the body-i and (x, y) axes
        # mass[None, :, None] has shape (1, N, 1)
        accel = self.G * np.sum(
            self.mass[None, :, None] * diff * inv_dist_cube[:, :, None],
            axis=1
        )  # (N, 2)

        return accel

    def step_euler(self, dt):
        """
        Forward Euler integration. Simple but energy drifts
        (always grows or always decays). We keep this around as a deliberately
        bad baseline to compare against leapfrog and the neural network.
        """
        accel = self.compute_accelerations()
        self.pos = self.pos + self.vel * dt
        self.vel = self.vel + accel * dt

    def step_leapfrog(self, dt):
        """
        Velocity Verlet (leapfrog) integration. Symplectic, so it conserves
        energy on average over long times. This is our "ground truth" against
        which we'll compare the neural network.

        The pattern is kick-drift-kick:
            v_{n+1/2} = v_n + a_n * dt/2       (kick)
            x_{n+1}   = x_n + v_{n+1/2} * dt   (drift)
            a_{n+1}   = a(x_{n+1})             (recompute forces)
            v_{n+1}   = v_{n+1/2} + a_{n+1} * dt/2  (kick)
        """
        accel = self.compute_accelerations()
        self.vel = self.vel + 0.5 * accel * dt           # half kick
        self.pos = self.pos + self.vel * dt              # full drift using the velocity at the middle of the time step
        accel_new = self.compute_accelerations()
        self.vel = self.vel + 0.5 * accel_new * dt       # half kick



    # diagnostics to make sure energy and momentum are conserved
    def kinetic_energy(self):
        # KE = sum_i 0.5 * m_i * |v_i|^2
        v_sq = np.sum(self.vel ** 2, axis=-1)            # (N,)
        return 0.5 * np.sum(self.mass * v_sq)

    def potential_energy(self):
        # U = -sum_{i<j} G * m_i * m_j / r_ij
        diff = self.pos[None, :, :] - self.pos[:, None, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + self.softening ** 2)
        np.fill_diagonal(dist, np.inf)  # so 1/r_ii = 0
        m_outer = self.mass[:, None] * self.mass[None, :]
        # the 0.5 corrects for double-counting in the sum over (i, j)
        return -0.5 * self.G * np.sum(m_outer / dist)

    def total_energy(self):
        return self.kinetic_energy() + self.potential_energy()

    def total_momentum(self):
        # p_total = sum_i m_i * v_i, returns shape (2,)
        return np.sum(self.mass[:, None] * self.vel, axis=0)

    # serialization helpers (we'll need these for the NN later)

    def state_vector(self):
        """
        Flatten current state into a 1D array of length 4*N:
            [x_0, y_0, x_1, y_1, ..., vx_0, vy_0, vx_1, vy_1, ...]
        This is the format we'll feed to the neural network.
        """
        return np.concatenate([self.pos.flatten(), self.vel.flatten()])

    @classmethod
    def from_state_vector(cls, state, masses, **kwargs):
        """Inverse of state_vector. Builds a System from a flat array."""
        N = len(masses)
        assert state.shape == (4 * N,)
        pos = state[:2 * N].reshape(N, 2)
        vel = state[2 * N:].reshape(N, 2)
        return cls(pos, vel, masses, **kwargs)


# ---- helpers for generating initial conditions ----

def figure_eight_initial_conditions():
    """
    The famous figure-8 choreography (Chenciner & Montgomery 2000) - discovered by Carles Símo
    Three equal-mass bodies in a stable closed orbit shaped like an 8.
    Great for testing because it's stable AND visually striking

    From - https://perso.imcce.fr/alain-chenciner/huit.pdf
    """
    pos = np.array([
        [ 0.97000436, -0.24308753],
        [-0.97000436,  0.24308753],
        [ 0.0,         0.0       ],
    ])
    vel = np.array([
        [ 0.466203685,  0.43236573],
        [ 0.466203685,  0.43236573],
        [-0.93240737,  -0.86473146],
    ])
    masses = np.array([1.0, 1.0, 1.0]) #must be same masses
    return pos, vel, masses


def random_three_body(seed=None, scale=1.0, vel_scale=0.5):
    """
    Random initial conditions for a 3-body system. Used for generating
    training data.

    We subtract the mean velocity so the center of mass doesn't drift,
    which keeps the system roughly centered for visualization and reduces
    one dimension of variability the network has to learn.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-scale, scale, size=(3, 2))
    vel = rng.uniform(-vel_scale, vel_scale, size=(3, 2))
    vel = vel - vel.mean(axis=0, keepdims=True)
    masses = np.ones(3)
    return pos, vel, masses


# quick self-test, runs when you do `python -m nbody.system`

if __name__ == "__main__":
    pos, vel, mass = figure_eight_initial_conditions()
    sys = System(pos, vel, mass)

    E0 = sys.total_energy()
    p0 = sys.total_momentum()
    print(f"Initial energy:   {E0:.6f}")
    print(f"Initial momentum: {p0}")

    dt = 0.001
    n_steps = 5000
    for _ in range(n_steps):
        sys.step_leapfrog(dt)

    E_final = sys.total_energy()
    p_final = sys.total_momentum()
    print(f"Final energy:     {E_final:.6f}")
    print(f"Energy drift:     {(E_final - E0) / abs(E0):.2e}")
    print(f"Final momentum:   {p_final}")
    print(f"Momentum drift:   {p_final - p0}")


"""
After running, this comes in the console:

Initial energy:   -1.287141
Initial momentum: [0. 0.]
Final energy:     -1.287141
Energy drift:     -2.36e-07
Final momentum:   [ 4.05231404e-15 -5.60662627e-15]
Momentum drift:   [ 4.05231404e-15 -5.60662627e-15]

Very low drift!!!!!


"""