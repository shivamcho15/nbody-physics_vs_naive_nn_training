# nbody/train.py
"""
Training loops for the two models. Loads a dataset, builds the right
PyTorch Dataset for each model, trains, and saves checkpoints.

Run from the project root:
    python -m nbody.train --model all --data data/train.npz --epochs 50
or just one model:
    python -m nbody.train --model force --epochs 30
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .data import load_dataset, split_dataset
from .models import NextStateMLP, ForceMLP, Normalizer


# Dataset classes (one per model, because each model has a different target)

class NextStateDataset(Dataset):
    """
    For NextStateMLP. Each sample is (state_t, state_{t+1}).
    We unroll all (n_steps) consecutive pairs from each trajectory.
    """

    def __init__(self, dataset_dict):
        # states shape: (n_traj, n_steps + 1, 4*N)
        states = dataset_dict["states"]
        # input is timesteps 0..T-1, target is timesteps 1..T
        self.inputs = torch.from_numpy(
            states[:, :-1, :].reshape(-1, states.shape[-1])
        ).float()
        self.targets = torch.from_numpy(
            states[:, 1:, :].reshape(-1, states.shape[-1])
        ).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


class ForceDataset(Dataset):
    """
    For ForceMLP. Input is positions only (2*N), target is acceleration (2*N).

    With softening=1e-3 in System, accelerations during near-encounters can
    spike to 10^3+. Those samples are out-of-distribution for what we want
    the network to learn (smooth regular gravity). Filter them out so the
    target distribution has a sane scale and the per-feature std used by the
    Normalizer isn't dominated by rare spikes.
    """

    def __init__(self, dataset_dict, max_accel=None):
        states = dataset_dict["states"]
        accels = dataset_dict["accels"]
        N = len(dataset_dict["masses"])
        positions = states[..., :2 * N]
        inputs = torch.from_numpy(positions.reshape(-1, 2 * N)).float()
        targets = torch.from_numpy(accels.reshape(-1, 2 * N)).float()

        if max_accel is not None:
            keep = targets.abs().amax(dim=1) <= max_accel
            n_kept = int(keep.sum())
            print(f"  ForceDataset: keeping {n_kept}/{len(keep)} samples "
                  f"(|a|_max <= {max_accel})")
            inputs = inputs[keep]
            targets = targets[keep]

        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


# Generic training loop (works for both models)

def train_model(model, train_loader, val_loader, epochs, lr, device,
                loss_fn=None, weight_decay=0.0):
    """
    Train a model and return (train_losses, val_losses, best_state_dict).

    loss_fn defaults to MSE. For acceleration targets (Force) we pass
    a Huber loss so close-encounter spikes (the tiny softening lets accels
    reach 1e3+) don't dominate the gradient.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = loss_fn if loss_fn is not None else nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(epochs):
        t0 = time.time()

        # training pass
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # validation pass
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = model(inputs)
                val_loss += criterion(preds, targets).item()
                n_val += 1
        val_loss = val_loss / n_val
        val_losses.append(val_loss)

        # save best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:3d}/{epochs}: train {train_loss:.4e}  "
              f"val {val_loss:.4e}  ({elapsed:.1f}s)")

    return train_losses, val_losses, best_state_dict


# Per-model training entry points

def train_nextstate(dataset_dict, epochs=50, batch_size=256, lr=1e-3,
                    hidden=256, n_layers=4, device="cpu",
                    save_dir="checkpoints"):
    train_data, val_data, _ = split_dataset(dataset_dict)
    train_ds = NextStateDataset(train_data)
    val_ds = NextStateDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    state_dim = train_ds.inputs.shape[1]
    model = NextStateMLP(state_dim, hidden=hidden, n_layers=n_layers)

    # build normalizer from training data only (don't peek at val/test)
    normalizer = Normalizer(train_ds.inputs).to(device)
    model.set_normalizer(normalizer)
    model.to(device)

    print(f"training NextStateMLP: {len(train_ds)} train, {len(val_ds)} val")
    train_losses, val_losses, best_state = train_model(
        model, train_loader, val_loader, epochs, lr, device,
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "nextstate.pt")
    torch.save({
        "state_dict": best_state,
        "normalizer_mean": normalizer.mean.cpu(),
        "normalizer_std": normalizer.std.cpu(),
        "state_dim": state_dim,
        "hidden": hidden,
        "n_layers": n_layers,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, save_path)
    print(f"saved best NextStateMLP to {save_path}")


def train_force(dataset_dict, epochs=50, batch_size=256, lr=1e-3,
                hidden=256, n_layers=4, device="cpu",
                save_dir="checkpoints", max_accel=50.0):
    train_data, val_data, _ = split_dataset(dataset_dict)
    train_ds = ForceDataset(train_data, max_accel=max_accel)
    val_ds = ForceDataset(val_data, max_accel=max_accel)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_bodies = len(dataset_dict["masses"])
    model = ForceMLP(n_bodies, hidden=hidden, n_layers=n_layers)

    pos_normalizer = Normalizer(train_ds.inputs).to(device)
    acc_normalizer = Normalizer(train_ds.targets).to(device)
    model.set_pos_normalizer(pos_normalizer)
    model.set_acc_normalizer(acc_normalizer)
    model.to(device)

    print(f"training ForceMLP: {len(train_ds)} train, {len(val_ds)} val")
    print(f"  acc target stats: mean={acc_normalizer.mean.abs().max().item():.3f} "
          f"std_max={acc_normalizer.std.max().item():.3f} "
          f"std_min={acc_normalizer.std.min().item():.3f}")

    # Custom loop: loss is computed in normalized-acceleration space so the
    # network only ever has to produce O(1) values, which it can.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_norm = acc_normalizer.normalize(targets)

            optimizer.zero_grad()
            preds_norm = model._net_out(inputs)
            loss = criterion(preds_norm, targets_norm)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_norm = acc_normalizer.normalize(targets)
                preds_norm = model._net_out(inputs)
                val_loss += criterion(preds_norm, targets_norm).item()
                n_val += 1
        val_loss /= n_val
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:3d}/{epochs}: train {train_loss:.4e}  "
              f"val {val_loss:.4e}  ({elapsed:.1f}s)")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "force.pt")
    torch.save({
        "state_dict": best_state_dict,
        "normalizer_mean": pos_normalizer.mean.cpu(),
        "normalizer_std": pos_normalizer.std.cpu(),
        "acc_normalizer_mean": acc_normalizer.mean.cpu(),
        "acc_normalizer_std": acc_normalizer.std.cpu(),
        "n_bodies": n_bodies,
        "hidden": hidden,
        "n_layers": n_layers,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, save_path)
    print(f"saved best ForceMLP to {save_path}")


# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nextstate", "force", "all"],
                        default="all", help="which model(s) to train")
    parser.add_argument("--data", default="data/test_dataset.npz",
                        help="path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    print(f"loading dataset from {args.data}")
    dataset = load_dataset(args.data)
    print(f"  {dataset['n_trajectories']} trajectories, "
          f"{dataset['n_steps']} steps each, dt={dataset['dt']}")
    print(f"  device: {args.device}")

    common_kwargs = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden=args.hidden, n_layers=args.n_layers,
        device=args.device, save_dir=args.save_dir,
    )

    if args.model in ("nextstate", "all"):
        print("\n" + "=" * 60)
        print("Model 1: NextStateMLP")
        print("=" * 60)
        train_nextstate(dataset, **common_kwargs)

    if args.model in ("force", "all"):
        print("\n" + "=" * 60)
        print("Model 2: ForceMLP")
        print("=" * 60)
        train_force(dataset, **common_kwargs)


if __name__ == "__main__":
    main()

"""
OUTPUT:
loading dataset from data/train.npz
  137 trajectories, 500 steps each, dt=0.01
  device: cpu

============================================================
Model 1: NextStateMLP
============================================================
training NextStateMLP: 54500 train, 6500 val
  epoch   1/50: train 8.6090e-03  val 1.8633e-02  (0.7s)
  epoch   2/50: train 7.3691e-03  val 1.5992e-02  (0.6s)
  epoch   3/50: train 6.5258e-03  val 1.6826e-02  (0.6s)
  epoch   4/50: train 5.8931e-03  val 1.5255e-02  (0.6s)
  epoch   5/50: train 5.0878e-03  val 1.5420e-02  (0.6s)
  epoch   6/50: train 4.5871e-03  val 1.5192e-02  (0.6s)
  epoch   7/50: train 4.0413e-03  val 1.5196e-02  (0.6s)
  epoch   8/50: train 3.5113e-03  val 1.4872e-02  (0.6s)
  epoch   9/50: train 3.0918e-03  val 1.4591e-02  (0.6s)
  epoch  10/50: train 2.7816e-03  val 1.4298e-02  (0.6s)
  epoch  11/50: train 2.3632e-03  val 1.4674e-02  (0.6s)
  epoch  12/50: train 2.1974e-03  val 1.4402e-02  (0.6s)
  epoch  13/50: train 1.8699e-03  val 1.5936e-02  (0.6s)
  epoch  14/50: train 1.7420e-03  val 1.5358e-02  (0.6s)
  epoch  15/50: train 1.5754e-03  val 1.4873e-02  (0.6s)
  epoch  16/50: train 1.4045e-03  val 1.4765e-02  (0.6s)
  epoch  17/50: train 1.2813e-03  val 1.5375e-02  (0.6s)
  epoch  18/50: train 1.1372e-03  val 1.5364e-02  (0.6s)
  epoch  19/50: train 1.1580e-03  val 1.5250e-02  (0.6s)
  epoch  20/50: train 1.0327e-03  val 1.5115e-02  (0.6s)
  epoch  21/50: train 9.8272e-04  val 1.5100e-02  (0.7s)
  epoch  22/50: train 9.8546e-04  val 1.5160e-02  (0.6s)
  epoch  23/50: train 9.6186e-04  val 1.5004e-02  (0.6s)
  epoch  24/50: train 8.3785e-04  val 1.6006e-02  (0.6s)
  epoch  25/50: train 8.5379e-04  val 1.5135e-02  (0.6s)
  epoch  26/50: train 7.4705e-04  val 1.5626e-02  (0.6s)
  epoch  27/50: train 7.5051e-04  val 1.5584e-02  (0.6s)
  epoch  28/50: train 8.5172e-04  val 1.5144e-02  (0.6s)
  epoch  29/50: train 6.9673e-04  val 1.6097e-02  (0.6s)
  epoch  30/50: train 6.6662e-04  val 1.5251e-02  (0.6s)
  epoch  31/50: train 6.8347e-04  val 1.5463e-02  (0.6s)
  epoch  32/50: train 6.8516e-04  val 1.5521e-02  (0.6s)
  epoch  33/50: train 6.8021e-04  val 1.5164e-02  (0.6s)
  epoch  34/50: train 6.3015e-04  val 1.4940e-02  (0.6s)
  epoch  35/50: train 6.0435e-04  val 1.4518e-02  (0.6s)
  epoch  36/50: train 6.3212e-04  val 1.4831e-02  (0.6s)
  epoch  37/50: train 6.2841e-04  val 1.5974e-02  (0.6s)
  epoch  38/50: train 5.4780e-04  val 1.5538e-02  (0.6s)
  epoch  39/50: train 5.4401e-04  val 1.5290e-02  (0.6s)
  epoch  40/50: train 5.0584e-04  val 1.5376e-02  (0.6s)
  epoch  41/50: train 5.1471e-04  val 1.5347e-02  (0.6s)
  epoch  42/50: train 5.1821e-04  val 1.4817e-02  (0.6s)
  epoch  43/50: train 5.1658e-04  val 1.5459e-02  (0.7s)
  epoch  44/50: train 5.3310e-04  val 1.5226e-02  (0.6s)
  epoch  45/50: train 4.7799e-04  val 1.5653e-02  (0.6s)
  epoch  46/50: train 5.0743e-04  val 1.5117e-02  (0.6s)
  epoch  47/50: train 5.4692e-04  val 1.4729e-02  (0.6s)
  epoch  48/50: train 5.5459e-04  val 1.4827e-02  (0.6s)
  epoch  49/50: train 4.9940e-04  val 1.5432e-02  (0.7s)
  epoch  50/50: train 4.6010e-04  val 1.5425e-02  (0.6s)
saved best NextStateMLP to checkpoints/nextstate.pt

============================================================
Model 2: ForceMLP
============================================================
training ForceMLP: 54609 train, 6513 val
  epoch   1/50: train 2.4665e+02  val 4.8303e+02  (0.6s)
  epoch   2/50: train 2.3740e+02  val 4.6756e+02  (0.7s)
  epoch   3/50: train 2.1577e+02  val 4.3310e+02  (0.6s)
  epoch   4/50: train 1.9365e+02  val 4.1046e+02  (0.6s)
  epoch   5/50: train 1.7175e+02  val 3.9243e+02  (0.7s)
  epoch   6/50: train 1.5436e+02  val 3.7718e+02  (0.6s)
  epoch   7/50: train 1.3683e+02  val 3.4881e+02  (0.6s)
  epoch   8/50: train 1.2677e+02  val 3.3515e+02  (0.6s)
  epoch   9/50: train 1.1252e+02  val 3.2199e+02  (0.6s)
  epoch  10/50: train 1.0078e+02  val 3.1853e+02  (0.6s)
  epoch  11/50: train 9.2794e+01  val 2.9878e+02  (0.6s)
  epoch  12/50: train 8.3894e+01  val 2.7923e+02  (0.6s)
  epoch  13/50: train 7.6109e+01  val 2.8248e+02  (0.6s)
  epoch  14/50: train 7.0660e+01  val 2.4662e+02  (0.7s)
  epoch  15/50: train 6.7769e+01  val 2.4598e+02  (0.7s)
  epoch  16/50: train 5.9141e+01  val 2.4033e+02  (0.6s)
  epoch  17/50: train 5.4093e+01  val 2.1624e+02  (0.6s)
  epoch  18/50: train 4.9395e+01  val 1.9477e+02  (0.6s)
  epoch  19/50: train 4.8851e+01  val 2.0039e+02  (0.7s)
  epoch  20/50: train 4.2760e+01  val 1.9634e+02  (0.6s)
  epoch  21/50: train 4.3540e+01  val 2.0406e+02  (0.6s)
  epoch  22/50: train 3.9562e+01  val 1.9036e+02  (0.6s)
  epoch  23/50: train 3.6067e+01  val 2.0022e+02  (0.6s)
  epoch  24/50: train 3.5551e+01  val 1.8107e+02  (0.6s)
  epoch  25/50: train 3.2831e+01  val 1.6066e+02  (0.7s)
  epoch  26/50: train 3.5222e+01  val 1.7238e+02  (0.9s)
  epoch  27/50: train 3.0955e+01  val 1.8103e+02  (0.7s)
  epoch  28/50: train 2.7727e+01  val 1.5912e+02  (0.7s)
  epoch  29/50: train 2.8395e+01  val 1.5280e+02  (0.7s)
  epoch  30/50: train 3.1434e+01  val 1.5366e+02  (0.6s)
  epoch  31/50: train 3.1184e+01  val 1.6321e+02  (0.6s)
  epoch  32/50: train 2.8104e+01  val 1.4693e+02  (0.6s)
  epoch  33/50: train 2.2215e+01  val 1.7815e+02  (0.7s)
  epoch  34/50: train 2.2342e+01  val 1.6334e+02  (0.6s)
  epoch  35/50: train 2.4692e+01  val 1.5797e+02  (0.6s)
  epoch  36/50: train 2.4152e+01  val 1.4522e+02  (0.6s)
  epoch  37/50: train 2.3114e+01  val 1.6157e+02  (0.6s)
  epoch  38/50: train 2.0534e+01  val 1.5271e+02  (0.6s)
  epoch  39/50: train 2.3633e+01  val 1.6315e+02  (0.6s)
  epoch  40/50: train 1.8376e+01  val 1.4557e+02  (0.6s)
  epoch  41/50: train 2.1832e+01  val 1.5504e+02  (0.6s)
  epoch  42/50: train 2.0531e+01  val 1.5368e+02  (0.6s)
  epoch  43/50: train 1.7808e+01  val 1.5265e+02  (0.6s)
  epoch  44/50: train 1.7805e+01  val 1.6328e+02  (0.6s)
  epoch  45/50: train 2.0334e+01  val 1.3525e+02  (0.6s)
  epoch  46/50: train 1.7421e+01  val 1.4744e+02  (0.7s)
  epoch  47/50: train 1.8317e+01  val 1.5817e+02  (0.6s)
  epoch  48/50: train 2.4468e+01  val 1.5382e+02  (0.6s)
  epoch  49/50: train 1.6564e+01  val 1.1853e+02  (0.6s)
  epoch  50/50: train 1.6788e+01  val 1.8084e+02  (0.6s)
saved best ForceMLP to checkpoints/force.pt

"""
