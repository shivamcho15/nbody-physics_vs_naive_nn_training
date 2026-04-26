# nbody/models.py
"""
Neural network models for predicting N-body dynamics.

Two models, ordered by how much physics structure they bake in:
    1. NextStateMLP - predicts the next state directly (no physics)
    2. ForceMLP     - predicts accelerations, then uses leapfrog (some physics)
"""

import torch
import torch.nn as nn
import numpy as np


def _make_mlp(in_dim, out_dim, hidden=256, n_layers=4, activation=nn.ReLU):
    """Tiny helper: build a fully connected MLP. All three models use this."""
    layers = [nn.Linear(in_dim, hidden), activation()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(activation())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# Normalizer (for all models)

class Normalizer:
    """
    Standardize features to mean=0, std=1. We store the statistics and use
    them at both train and inference time.

    Positions are O(1) but velocities are O(0.5) and
    accelerations are O(10) in our setup. MLPs train badly on inputs with
    very different scales because the gradients become unbalanced across
    features. Normalizing fixes that.
    """

    def __init__(self, data):
        # data: (N_samples, dim) tensor or numpy array
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-8  # avoid divide by zero on dead features

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


# Model 1: NextStateMLP (the dumb baseline)

class NextStateMLP(nn.Module):
    """
    Take state at time t, predict state at time t+dt. We actually predict
    the delta (state_{t+1} - state_t) and add it to the input. For small
    dt the delta is close to zero, which is way easier to learn than full
    state values that can be anywhere in space.

    No physics structure baked in. This is the throw an
    MLP at it baseline that i expect to fail
    """

    def __init__(self, state_dim, hidden=256, n_layers=4):
        super().__init__()
        self.state_dim = state_dim
        self.net = _make_mlp(state_dim, state_dim, hidden, n_layers)
        self.normalizer = None  # set externally after dataset is loaded

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, state):
        # state: (batch, state_dim)
        if self.normalizer is not None:
            x = self.normalizer.normalize(state)
        else:
            x = state
        delta = self.net(x)
        return state + delta  # residual in physical space

    @torch.no_grad()
    def rollout(self, initial_state, n_steps):
        """
        Autoregressive: feed each prediction back in as the next input.
        This is where errors compound. Watching the rollout error grow over
        time is the main result we're after.

        initial_state: (state_dim,)
        Returns: (n_steps + 1, state_dim)
        """
        states = [initial_state]
        state = initial_state.unsqueeze(0)  # add batch dim
        for _ in range(n_steps):
            state = self.forward(state)
            states.append(state.squeeze(0))
        return torch.stack(states)


# Model 2: ForceMLP (predicts accelerations, integrates with leapfrog)

class ForceMLP(nn.Module):
    """
    MLP that predicts accelerations from positions only, then uses the
    analytical leapfrog integrator to step forward.

    Two pieces of physics baked in:
        1. Accelerations depend on positions, NOT velocities (Newton's law).
           So the input is (2*N,), not (4*N,).
        2. The leapfrog integrator is symplectic, so it conserves energy
           on average. The NN only has to learn the hard part (the forces).

    This should beat NextStateMLP by a wide margin on long rollouts.
    The contrast between the two is the core empirical result of the project.
    """

    def __init__(self, n_bodies, hidden=256, n_layers=4):
        super().__init__()
        self.n_bodies = n_bodies
        self.pos_dim = 2 * n_bodies
        self.net = _make_mlp(self.pos_dim, self.pos_dim, hidden, n_layers)
        self.pos_normalizer = None
        self.acc_normalizer = None

    def set_pos_normalizer(self, normalizer):
        self.pos_normalizer = normalizer

    def set_acc_normalizer(self, normalizer):
        # Normalizing the acceleration targets is the single biggest fix here.
        # Without it the network has to produce huge raw values (accels can be
        # O(10^2) near close passes) which a small-init linear head can't reach.
        self.acc_normalizer = normalizer

    def _net_out(self, positions):
        """Raw network output. Lives in normalized-acceleration space when
        acc_normalizer is set, which is what the training loss compares against."""
        if self.pos_normalizer is not None:
            x = self.pos_normalizer.normalize(positions)
        else:
            x = positions
        return self.net(x)

    def forward(self, positions):
        """Predict accelerations in physical units (used by rollout)."""
        out = self._net_out(positions)
        if self.acc_normalizer is not None:
            return self.acc_normalizer.denormalize(out)
        return out

    @torch.no_grad()
    def rollout(self, initial_state, n_steps, dt):
        """
        Leapfrog using NN-predicted accelerations. Note that the
        integrator structure is hard-coded; only the force prediction is
        learned. That's the whole point.
        """
        N = self.n_bodies
        states = [initial_state]
        pos = initial_state[:2 * N].clone()
        vel = initial_state[2 * N:].clone()

        for _ in range(n_steps):
            # leapfrog kick-drift-kick (same pattern as System.step_leapfrog)
            a = self.forward(pos.unsqueeze(0)).squeeze(0)
            vel = vel + 0.5 * a * dt
            pos = pos + vel * dt
            a_new = self.forward(pos.unsqueeze(0)).squeeze(0)
            vel = vel + 0.5 * a_new * dt
            states.append(torch.cat([pos, vel]))

        return torch.stack(states)


# Sanity test

if __name__ == "__main__":
    n_bodies = 3
    state_dim = 4 * n_bodies
    states = torch.randn(8, state_dim)

    print("=== NextStateMLP ===")
    m1 = NextStateMLP(state_dim)
    out = m1(states)
    print(f"  forward: {states.shape} -> {out.shape}")
    assert out.shape == states.shape
    rollout = m1.rollout(states[0], n_steps=10)
    print(f"  rollout: {rollout.shape}")

    print("=== ForceMLP ===")
    m2 = ForceMLP(n_bodies)
    positions = states[:, :2 * n_bodies]
    accel = m2(positions)
    print(f"  forward: {positions.shape} -> {accel.shape}")
    rollout2 = m2.rollout(states[0], n_steps=10, dt=0.01)
    print(f"  rollout: {rollout2.shape}")

    print("\nall models ok")


"""
OUTPUT:

=== NextStateMLP ===
  forward: torch.Size([8, 12]) -> torch.Size([8, 12])
  rollout: torch.Size([11, 12])
=== ForceMLP ===
  forward: torch.Size([8, 6]) -> torch.Size([8, 6])
  rollout: torch.Size([11, 12])

all models ok

Models work!!!
"""