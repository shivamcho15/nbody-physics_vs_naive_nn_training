# nbody/data.py
"""
Trajectory generation for training. Runs the System simulation forward in
time many times with random initial conditions, then saves the resulting
arrays to disk so we can reuse the same dataset across experiments.
"""

import numpy as np
from tqdm import tqdm #for loading bars

from .system import System, random_three_body


def generate_trajectory(positions, velocities, masses, n_steps, dt,
                        record_accelerations=True):
    """
    Run a single trajectory with the leapfrog integrator. Returns:
        states: (n_steps + 1, 4*N) array, state at every timestep
        accels: (n_steps + 1, 2*N) array, accelerations at every timestep
                (used for training the force-prediction model later)
    """
    sys = System(positions, velocities, masses)
    N = sys.N

    states = np.zeros((n_steps + 1, 4 * N))
    accels = np.zeros((n_steps + 1, 2 * N)) if record_accelerations else None

    # record initial state
    states[0] = sys.state_vector()
    if record_accelerations:
        accels[0] = sys.compute_accelerations().flatten()

    # step forward and record at every timestep
    for t in range(n_steps):
        sys.step_leapfrog(dt)
        states[t + 1] = sys.state_vector()
        if record_accelerations:
            accels[t + 1] = sys.compute_accelerations().flatten()

    return states, accels


def generate_dataset(n_trajectories=1000, n_steps=500, dt=0.01,
                     seed=42, scale=1.0, vel_scale=0.5,
                     filter_unstable=True, energy_threshold=0.5):
    """
    Generate many random trajectories and stack them. Throws out trajectories
    where energy drifts a lot (those had a near-collision and are basically
    garbage for training).

    Returns a dict with keys: states, accels, masses, dt, n_trajectories, n_steps.
    """
    rng = np.random.default_rng(seed)
    # we generate spare seeds in case some trajectories fail the stability filter
    seeds = rng.integers(0, 2**32 - 1, size=n_trajectories * 3)

    all_states = []
    all_accels = []
    masses_ref = None
    n_kept = 0
    n_tried = 0

    pbar = tqdm(total=n_trajectories, desc="generating trajectories")
    while n_kept < n_trajectories and n_tried < len(seeds):
        ic_seed = int(seeds[n_tried])
        n_tried += 1

        pos, vel, masses = random_three_body(
            seed=ic_seed, scale=scale, vel_scale=vel_scale
        )
        if masses_ref is None:
            masses_ref = masses

        states, accels = generate_trajectory(pos, vel, masses, n_steps, dt)

        if filter_unstable:
            # check energy drift over the whole trajectory. If it's huge, the trajectory probably had a close encounter and the integrator blew up. Skip it and try a different IC.
            sys_init = System(pos, vel, masses)
            E0 = sys_init.total_energy()
            sys_final = System.from_state_vector(states[-1], masses)
            E_final = sys_final.total_energy()
            drift = abs(E_final - E0) / (abs(E0) + 1e-10)
            if drift > energy_threshold:
                continue

        all_states.append(states)
        all_accels.append(accels)
        n_kept += 1
        pbar.update(1)
    pbar.close()

    if n_kept < n_trajectories:
        print(f"warning: only kept {n_kept}/{n_trajectories} stable trajectories. "
              f"try lowering vel_scale or raising scale to reduce close encounters.")

    return {
        "states": np.array(all_states),       # (n_traj, n_steps+1, 4*N)
        "accels": np.array(all_accels),       # (n_traj, n_steps+1, 2*N)
        "masses": masses_ref,                 # (N,)
        "dt": dt,
        "n_trajectories": n_kept,
        "n_steps": n_steps,
    }


def save_dataset(dataset, path):
    """Save to a compressed .npz file."""
    np.savez_compressed(
        path,
        states=dataset["states"],
        accels=dataset["accels"],
        masses=dataset["masses"],
        dt=dataset["dt"],
    )
    print(f"saved {dataset['n_trajectories']} trajectories to {path}")


def load_dataset(path):
    """Load a dataset from a .npz file."""
    data = np.load(path)
    return {
        "states": data["states"],
        "accels": data["accels"],
        "masses": data["masses"],
        "dt": float(data["dt"]),
        "n_trajectories": data["states"].shape[0],
        "n_steps": data["states"].shape[1] - 1,
    }


def split_dataset(dataset, train_frac=0.8, val_frac=0.1, seed=0):
    """
    Split into train/val/test by trajectory (NOT by timestep).
    This is important: if we mixed timesteps from the same trajectory across
    splits, the network could memorize specific trajectories and look much
    better than it really is. Splitting by trajectory means the test set is
    truly unseen initial conditions.
    """
    n = dataset["n_trajectories"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def subset(indices):
        return {
            "states": dataset["states"][indices],
            "accels": dataset["accels"][indices],
            "masses": dataset["masses"],
            "dt": dataset["dt"],
            "n_trajectories": len(indices),
            "n_steps": dataset["n_steps"],
        }

    return subset(train_idx), subset(val_idx), subset(test_idx)


if __name__ == "__main__":
    # smoke test: generate a small dataset, split, save, reload
    import os

    print("generating small test dataset...")
    dataset = generate_dataset(n_trajectories=20, n_steps=200, dt=0.01, seed=42)

    print(f"states shape: {dataset['states'].shape}")
    print(f"accels shape: {dataset['accels'].shape}")

    train, val, test = split_dataset(dataset)
    print(f"train: {train['n_trajectories']}, val: {val['n_trajectories']}, "
          f"test: {test['n_trajectories']}")

    os.makedirs("data", exist_ok=True)
    save_dataset(dataset, "data/test_dataset.npz")
    reloaded = load_dataset("data/test_dataset.npz")
    assert np.allclose(reloaded["states"], dataset["states"])
    print("save/load round-trip ok")

"""
OUTPUT:
warning: only kept 5/20 stable trajectories. try lowering vel_scale or raising scale to reduce close encounters.
states shape: (5, 201, 12)
accels shape: (5, 201, 6)
train: 4, val: 0, test: 1
saved 5 trajectories to data/test_dataset.npz
save/load round-trip ok

"""