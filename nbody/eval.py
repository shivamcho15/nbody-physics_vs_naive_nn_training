# nbody/eval.py
"""
Evaluation experiments: load trained models, run rollouts on a held-out
test trajectory, compute metrics, and produce all the plots that go in
the README.

Run from project root:
    python -m nbody.eval --data data/train.npz
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from .system import System
from .data import load_dataset, split_dataset
from .models import NextStateMLP, ForceMLP, Normalizer


# ----------------------------------------------------------------------
# Loading trained models
# ----------------------------------------------------------------------

def _make_normalizer(mean, std, device):
    """
    Reconstruct a Normalizer from saved mean/std without re-running __init__
    (which would need the original training data).
    """
    norm = Normalizer.__new__(Normalizer)
    norm.mean = mean.to(device)
    norm.std = std.to(device)
    return norm


def load_nextstate(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NextStateMLP(ckpt["state_dim"], hidden=ckpt["hidden"],
                          n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.set_normalizer(_make_normalizer(ckpt["normalizer_mean"],
                                          ckpt["normalizer_std"], device))
    model.to(device).eval()
    return model, ckpt


def load_force(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ForceMLP(ckpt["n_bodies"], hidden=ckpt["hidden"],
                     n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.set_pos_normalizer(_make_normalizer(ckpt["normalizer_mean"],
                                              ckpt["normalizer_std"], device))
    if "acc_normalizer_mean" in ckpt:
        model.set_acc_normalizer(_make_normalizer(ckpt["acc_normalizer_mean"],
                                                  ckpt["acc_normalizer_std"], device))
    model.to(device).eval()
    return model, ckpt


# ----------------------------------------------------------------------
# Generate ground-truth and model rollouts
# ----------------------------------------------------------------------

def true_trajectory(pos, vel, masses, n_steps, dt):
    """Run leapfrog as the ground truth."""
    sys = System(pos, vel, masses)
    states = np.zeros((n_steps + 1, 4 * sys.N))
    states[0] = sys.state_vector()
    for t in range(n_steps):
        sys.step_leapfrog(dt)
        states[t + 1] = sys.state_vector()
    return states


def euler_trajectory(pos, vel, masses, n_steps, dt):
    """Forward Euler as a deliberately bad classical baseline."""
    sys = System(pos, vel, masses)
    states = np.zeros((n_steps + 1, 4 * sys.N))
    states[0] = sys.state_vector()
    for t in range(n_steps):
        sys.step_euler(dt)
        states[t + 1] = sys.state_vector()
    return states


def model_rollout(model, model_name, initial_state, n_steps, dt, device="cpu"):
    """Wrap the model.rollout call (signatures differ slightly)."""
    initial = torch.from_numpy(initial_state).float().to(device)
    if model_name == "nextstate":
        traj = model.rollout(initial, n_steps)
    else:
        traj = model.rollout(initial, n_steps, dt)
    return traj.detach().cpu().numpy()


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def position_error(true_states, pred_states, N):
    """RMS position error per timestep, averaged over bodies. Returns (T,)."""
    true_pos = true_states[:, :2 * N].reshape(-1, N, 2)
    pred_pos = pred_states[:, :2 * N].reshape(-1, N, 2)
    diff = pred_pos - true_pos
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=-1), axis=-1))


def energy_over_time(states, masses):
    """Total energy at each timestep. Returns (T,)."""
    energies = np.zeros(len(states))
    for t in range(len(states)):
        sys = System.from_state_vector(states[t], masses)
        energies[t] = sys.total_energy()
    return energies


# ----------------------------------------------------------------------
# Plot 1: training loss curves
# ----------------------------------------------------------------------

def plot_loss_curves(checkpoints, save_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"nextstate": "C0", "force": "C1"}
    labels = {"nextstate": "NextStateMLP", "force": "ForceMLP"}

    for name, ckpt in checkpoints.items():
        if ckpt is None:
            continue
        epochs = range(1, len(ckpt["train_losses"]) + 1)
        ax.plot(epochs, ckpt["train_losses"], color=colors[name],
                linestyle="-", label=f"{labels[name]} (train)", alpha=0.8)
        ax.plot(epochs, ckpt["val_losses"], color=colors[name],
                linestyle="--", label=f"{labels[name]} (val)", alpha=0.8)

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title("Training and validation loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
# Plot 2: trajectory comparison (eye candy for the README)
# ----------------------------------------------------------------------

def plot_trajectory_comparison(true_traj, model_trajs, N, save_dir):
    n_panels = 1 + len(model_trajs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4),
                              sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    body_colors = ["C0", "C1", "C2", "C3", "C4"][:N]

    def draw(ax, traj, title):
        pos = traj[:, :2 * N].reshape(-1, N, 2)
        for b in range(N):
            ax.plot(pos[:, b, 0], pos[:, b, 1], color=body_colors[b], lw=1)
            ax.plot(pos[0, b, 0], pos[0, b, 1], "o", color=body_colors[b], ms=6)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    draw(axes[0], true_traj, "Ground truth (leapfrog)")
    for ax, (name, traj) in zip(axes[1:], model_trajs.items()):
        draw(ax, traj, name)

    fig.tight_layout()
    path = os.path.join(save_dir, "trajectory_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
# Plot 3: position error vs time
# ----------------------------------------------------------------------

def plot_position_error(true_traj, model_trajs, dt, N, save_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    times = np.arange(len(true_traj)) * dt
    for name, traj in model_trajs.items():
        err = position_error(true_traj, traj, N)
        ax.plot(times, err, label=name, lw=1.5)
    ax.set_xlabel("time")
    ax.set_ylabel("RMS position error (averaged over bodies)")
    ax.set_yscale("log")
    ax.set_title("Position error growth during autoregressive rollout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "position_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
# Plot 4: energy drift vs time (the main physics result)
# ----------------------------------------------------------------------

def plot_energy_drift(true_traj, model_trajs, dt, masses, save_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    times = np.arange(len(true_traj)) * dt

    E0 = System.from_state_vector(true_traj[0], masses).total_energy()

    # leapfrog as the gold-standard reference
    E_true = energy_over_time(true_traj, masses)
    ax.plot(times, (E_true - E0) / abs(E0),
            label="leapfrog (truth)", color="black", lw=1.5)

    for name, traj in model_trajs.items():
        E = energy_over_time(traj, masses)
        ax.plot(times, (E - E0) / abs(E0), label=name, lw=1.5)

    ax.set_xlabel("time")
    ax.set_ylabel("(E - E0) / |E0|")
    ax.set_title("Relative energy drift over the rollout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "energy_drift.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
# Plot 5: speed vs accuracy
# ----------------------------------------------------------------------

def benchmark_speed(fn, n_trials=100):
    """Average wall-clock for one call."""
    for _ in range(5):  # warmup
        fn()
    t0 = time.time()
    for _ in range(n_trials):
        fn()
    return (time.time() - t0) / n_trials


def plot_speed_vs_accuracy(true_traj, model_trajs, speeds, N, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, traj in model_trajs.items():
        err = position_error(true_traj, traj, N)[-1]
        ax.scatter(speeds[name] * 1000, err, s=80)
        ax.annotate(name, (speeds[name] * 1000, err),
                    xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("time per step (ms)")
    ax.set_ylabel("final position error after rollout")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Speed vs. accuracy tradeoff")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(save_dir, "speed_vs_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
# Main: orchestrate everything
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.npz")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--save_dir", default="figures")
    parser.add_argument("--n_steps", type=int, default=500,
                        help="rollout length in timesteps")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- load data, pick a held-out test trajectory ----
    print(f"loading dataset from {args.data}")
    dataset = load_dataset(args.data)
    masses = dataset["masses"]
    dt = dataset["dt"]
    N = len(masses)

    _, _, test_data = split_dataset(dataset)
    print(f"  rollout: {args.n_steps} steps, dt={dt}, N={N} bodies")

    # search for a test trajectory whose leapfrog ground truth is stable
    # over the requested rollout length (no close encounters)
    print("finding a stable test trajectory...")
    best_idx, best_drift, best_traj = None, float("inf"), None
    for idx in range(min(50, test_data["n_trajectories"])):
        init = test_data["states"][idx, 0]
        p = init[:2 * N].reshape(N, 2)
        v = init[2 * N:].reshape(N, 2)
        traj = true_trajectory(p, v, masses, args.n_steps, dt)
        E0 = System.from_state_vector(traj[0], masses).total_energy()
        Ef = System.from_state_vector(traj[-1], masses).total_energy()
        drift = abs(Ef - E0) / abs(E0)
        if drift < best_drift:
            best_idx, best_drift, best_traj = idx, drift, traj
        if drift < 0.01:  # already very stable, no need to keep searching
            break
    print(f"  picked trajectory {best_idx} (leapfrog drift: {best_drift:.2e})")

    initial_state = test_data["states"][best_idx, 0]
    pos = initial_state[:2 * N].reshape(N, 2)
    vel = initial_state[2 * N:].reshape(N, 2)
    true_traj = best_traj
    print("running Euler (bad baseline)...")
    euler_traj = euler_trajectory(pos, vel, masses, args.n_steps, dt)

    # ---- load and roll out NN models ----
    model_trajs = {"Euler": euler_traj}
    checkpoints = {}

    print("loading and rolling out NextStateMLP...")
    m1, ck1 = load_nextstate(os.path.join(args.checkpoints, "nextstate.pt"), args.device)
    model_trajs["NextStateMLP"] = model_rollout(m1, "nextstate", initial_state,
                                                args.n_steps, dt, args.device)
    checkpoints["nextstate"] = ck1

    print("loading and rolling out ForceMLP...")
    m2, ck2 = load_force(os.path.join(args.checkpoints, "force.pt"), args.device)
    model_trajs["ForceMLP"] = model_rollout(m2, "force", initial_state,
                                            args.n_steps, dt, args.device)
    checkpoints["force"] = ck2

    # ---- generate all plots ----
    print("\nplots:")
    plot_loss_curves(checkpoints, args.save_dir)
    plot_trajectory_comparison(true_traj, model_trajs, N, args.save_dir)
    plot_position_error(true_traj, model_trajs, dt, N, args.save_dir)
    plot_energy_drift(true_traj, model_trajs, dt, masses, args.save_dir)

    # ---- speed benchmark ----
    print("\nbenchmarking step times...")
    speeds = {}

    sys_e = System(pos, vel, masses)
    speeds["Euler"] = benchmark_speed(lambda: sys_e.step_euler(dt))

    sys_lf = System(pos, vel, masses)
    leapfrog_speed = benchmark_speed(lambda: sys_lf.step_leapfrog(dt))

    initial_t = torch.from_numpy(initial_state).float().to(args.device)
    pos_t = initial_t[:2 * N].unsqueeze(0)
    speeds["NextStateMLP"] = benchmark_speed(lambda: m1(initial_t.unsqueeze(0)))
    speeds["ForceMLP"] = benchmark_speed(lambda: m2(pos_t))

    print(f"  Euler:        {speeds['Euler']*1000:.3f} ms")
    print(f"  leapfrog:     {leapfrog_speed*1000:.3f} ms (truth)")
    print(f"  NextStateMLP: {speeds['NextStateMLP']*1000:.3f} ms")
    print(f"  ForceMLP:     {speeds['ForceMLP']*1000:.3f} ms")

    plot_speed_vs_accuracy(true_traj, model_trajs, speeds, N, args.save_dir)

    # ---- summary numbers (for the README) ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    # Print position error at several checkpoints, not just the final step.
    # Chaotic 3-body trajectories saturate to ~the orbit scale by the end of
    # a long rollout, so the final-step number flattens out the gap between
    # models. Intermediate snapshots actually show which model is tracking.
    snapshots = [int(args.n_steps * f) for f in (0.1, 0.25, 0.5, 1.0)]
    print(f"Position error (RMS over bodies) at fractions of rollout:")
    header = "  " + " " * 16 + "  ".join(f"t={s*dt:5.2f}" for s in snapshots)
    print(header)
    for name, traj in model_trajs.items():
        err = position_error(true_traj, traj, N)
        cells = "  ".join(f"{err[s]:7.3e}" for s in snapshots)
        print(f"  {name:15s} {cells}")

    print(f"\nFinal relative energy drift:")
    E0 = System.from_state_vector(true_traj[0], masses).total_energy()
    drift_lf = (energy_over_time(true_traj, masses)[-1] - E0) / abs(E0)
    print(f"  leapfrog (truth): {drift_lf:.4e}")
    for name, traj in model_trajs.items():
        E_final = energy_over_time(traj, masses)[-1]
        drift = (E_final - E0) / abs(E0)
        print(f"  {name:15s}: {drift:.4e}")


if __name__ == "__main__":
    main()