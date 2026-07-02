"""
Run the community abundance ODE using the trained surrogate in place of LP.

Replicates the Euler simulation loop from mcsm_simulate.py but calls the
surrogate neural network instead of cooperative_tradeoff.  Starting conditions
are read directly from the LP trajectory file so both routes use identical
initial states.

Usage:
    python scripts/mcsm_ode_surrogate.py \\
        --checkpoint trained_models/mcsm_community_input-6_output-6_hidden-64.pt \\
        --lp-trajectories results/mcsm_trajectories.npz \\
        --output results/mcsm_surrogate_trajectories.npz \\
        --n-steps 80 \\
        --dt 0.1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN                                     # noqa: E402

MIN_ABUND = 1e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "mcsm_community_input-6_output-6_hidden-64.pt")
    p.add_argument("--lp-trajectories", type=Path,
                   default=REPO_ROOT / "results" / "mcsm_trajectories.npz")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "results" / "mcsm_surrogate_trajectories.npz")
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--dt",      type=float, default=0.1)
    return p.parse_args()


def euler_step(x, mu, dt):
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading surrogate checkpoint")
    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Metadata   : vman={metadata.get('vman')}, "
          f"condition={metadata.get('condition')}, "
          f"hidden_dim={metadata.get('hidden_dim')}")

    section("2. Loading LP trajectory starting conditions")
    lp = np.load(args.lp_trajectories, allow_pickle=True)
    taxa_ids      = list(lp["taxa_ids"])
    profile_names = list(lp["profile_names"])
    lp_trajs      = lp["trajectories"]   # (n_profiles, n_steps+1, n_taxa)
    n_taxa        = len(taxa_ids)
    n_profiles    = len(profile_names)
    print(f"  Taxa       : {taxa_ids}")
    print(f"  Profiles   : {profile_names}")
    print(f"  Simulating {n_profiles} trajectories × {args.n_steps} steps × dt={args.dt}h")

    section("3. Surrogate simulation loop")
    all_trajectories = []   # (n_profiles, n_steps+1, n_taxa)
    all_mu_trajs     = []   # (n_profiles, n_steps, n_taxa)
    all_comm_gr      = []   # (n_profiles, n_steps)  abundance-weighted mu

    t_total = time.time()
    for p, name in enumerate(profile_names):
        x0 = lp_trajs[p, 0, :].copy()
        traj    = np.full((args.n_steps + 1, n_taxa), np.nan)
        mu_traj = np.full((args.n_steps,     n_taxa), np.nan)
        gr_traj = np.full((args.n_steps,),           np.nan)
        traj[0] = x0
        x = x0.copy()

        print(f"\n  ── {name} ──")
        for step in range(args.n_steps):
            mu = surrogate_predict(model_nn, x_scaler, y_scaler, x).flatten()
            mu_traj[step] = mu
            gr_traj[step] = float(x @ mu)   # abundance-weighted community growth rate
            x = euler_step(x, mu, args.dt)
            traj[step + 1] = x

            if (step + 1) % 10 == 0:
                dominant = taxa_ids[np.argmax(x)]
                print(f"    step {step+1:>3}  t={(step+1)*args.dt:5.1f}h  "
                      f"comm_gr(proxy)={gr_traj[step]:.4f}  "
                      f"dominant={dominant} ({x.max():.3f})")

        final_x  = traj[-1]
        dominant = taxa_ids[np.argmax(final_x)]
        print(f"  Final state (t={args.n_steps*args.dt:.1f}h): "
              f"dominant={dominant} ({final_x.max():.4f})")

        all_trajectories.append(traj)
        all_mu_trajs.append(mu_traj)
        all_comm_gr.append(gr_traj)

    print(f"\n  Total wall time: {time.time()-t_total:.1f} s")

    section("4. Saving")
    np.savez_compressed(
        args.output,
        taxa_ids      = taxa_ids,
        profile_names = profile_names,
        trajectories  = np.stack(all_trajectories),   # (n_profiles, n_steps+1, n_taxa)
        mu_trajs      = np.stack(all_mu_trajs),       # (n_profiles, n_steps, n_taxa)
        gr_trajs      = np.stack(all_comm_gr),        # (n_profiles, n_steps)
        dt            = args.dt,
        n_steps       = args.n_steps,
        checkpoint    = str(args.checkpoint),
    )
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
