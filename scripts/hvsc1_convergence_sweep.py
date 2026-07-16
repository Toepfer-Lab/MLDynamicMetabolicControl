"""
Convergence-truncated LP + surrogate trajectory sweep, for browsing a large
pool of diverse 27D starting conditions (see
scripts/hvsc1_convergence_sweep_generate_ics.py).

For each starting abundance vector in a (possibly sharded) slice of the ICs
file, runs the SAME LP ODE loop as hvsc1_simulate.py -- Euler/replicator
update x_new = x*(1+mu*dt), renormalized -- but stops as soon as one taxon's
abundance crosses --convergence-threshold (default 0.99), same early-stop
logic hvsc1_simulate.py already uses. In production, real trajectories
converge in a handful of steps (5-55 observed, vs. the fixed n_steps=80
budget), so truncating on convergence lets many more starting points be
explored for the same CPLEX budget than a fixed-length sweep would.

Also runs the (near-zero-cost, no LP) surrogate ODE for the same starting
point with its own convergence-truncation, and saves ONE single-panel plot
per starting configuration (LP solid + surrogate dashed, one line per taxon)
into --plots-dir, for later browsing/picking interesting cases to discuss.

Also saves the full per-step (x, mu, gr) trajectory data for both LP and
surrogate, plus a flattened (X_steps, Y_steps) array of every (abundance,
growth_rate) pair visited along the LP trajectories in this shard -- since
cooperative_tradeoff is the same "ground truth" oracle used for training-data
generation, these are usable as extra ground-truth data if wanted later
(NOT auto-merged into data/hvsc1_training.npz -- these points are correlated
along a trajectory, not i.i.d., so folding them into training is a decision
to make separately, not something this script does on its own).

Stage: needs CPLEX (LP solves) -- run via
scripts/hvsc1_convergence_sweep_array.sh on the cluster, sharding a slice of
the fixed ICs file per SLURM array task (same deterministic-split pattern as
hvsc1_generate_data.py's --candidates-file mode).

Usage:
    python scripts/hvsc1_convergence_sweep.py \\
        --ics-file results/hvsc1_convergence_sweep_ics.npz \\
        --shard-index 0 --n-shards 20 \\
        --plots-dir plots/hvsc1_convergence_sweep \\
        --output results/hvsc1_convergence_sweep/shard00.npz
"""

import argparse
import sys
import time
from pathlib import Path

import cobra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from micom import load_pickle

cobra.Configuration.solver = "cplex"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN  # noqa: E402

MIN_ABUND = 1e-8


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ics-file", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_convergence_sweep_ics.npz")
    p.add_argument("--shard-index", type=int, default=None,
                   help="This task's index into an --n-shards-way deterministic split "
                        "of the ICs file (np.array_split over global indices).")
    p.add_argument("--n-shards", type=int, default=None,
                   help="Total number of shards (must match the SLURM --array size). "
                        "If omitted (with --shard-index also omitted), runs all ICs.")
    p.add_argument("--model-path", type=Path,
                   default=REPO_ROOT / "model" / "hvsc1_comm.pickle")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--convergence-threshold", type=float, default=0.99,
                   help="Stop a trajectory once the dominant taxon exceeds this "
                        "abundance -- same early-stop rule as hvsc1_simulate.py.")
    p.add_argument("--plots-dir", type=Path,
                   default=REPO_ROOT / "plots" / "hvsc1_convergence_sweep")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_convergence_sweep"
                           / "hvsc1_convergence_sweep_shard.npz")
    return p.parse_args()


def apply_medium_and_unlock(comm, default_medium):
    comm.medium = default_medium
    for rxn in comm.internal_exchanges:
        rxn.bounds = (-1000.0, 1000.0)


def solve(comm, abundance_vec, taxa_ids, default_medium, fraction):
    comm.set_abundance(pd.Series(dict(zip(taxa_ids, abundance_vec))))
    apply_medium_and_unlock(comm, default_medium)
    t0 = time.time()
    sol = comm.cooperative_tradeoff(fraction=fraction, pfba=False, fluxes=False)
    elapsed = time.time() - t0
    mu = sol.members.reindex(taxa_ids)["growth_rate"].values.astype(float)
    return mu, sol.status, float(sol.growth_rate), elapsed


def euler_step(x, mu, dt):
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


def run_lp_trajectory(comm, x0, taxa_ids, default_medium, fraction, n_steps, dt,
                      convergence_threshold):
    """Same early-stop Euler loop as hvsc1_simulate.py. Returns (traj, mu_traj,
    gr_traj, n_lp_calls) where traj has shape (n_steps+1, n_taxa) and the tail
    is filled with the converged state once convergence triggers."""
    n_taxa = len(taxa_ids)
    traj = np.full((n_steps + 1, n_taxa), np.nan)
    mu_traj = np.full((n_steps, n_taxa), np.nan)
    gr_traj = np.full((n_steps,), np.nan)
    traj[0] = x0.copy()
    x = x0.copy()

    last_mu, last_gr = np.zeros(n_taxa), 0.0
    n_lp_calls = 0
    for step in range(n_steps):
        if np.max(x) >= convergence_threshold:
            for rem in range(step, n_steps):
                traj[rem + 1] = x
                mu_traj[rem] = last_mu
                gr_traj[rem] = last_gr
            return traj, mu_traj, gr_traj, n_lp_calls

        mu, status, comm_gr, _ = solve(comm, x, taxa_ids, default_medium, fraction)
        n_lp_calls += 1
        if status != "optimal":
            mu, comm_gr = last_mu.copy(), last_gr
        else:
            last_mu, last_gr = mu.copy(), comm_gr

        mu_traj[step] = mu
        gr_traj[step] = comm_gr
        x = euler_step(x, mu, dt)
        traj[step + 1] = x

    return traj, mu_traj, gr_traj, n_lp_calls


def run_surrogate_trajectory(model_nn, x_scaler, y_scaler, x0, n_steps, dt,
                             convergence_threshold):
    """Same early-stop Euler loop, using the (cheap) surrogate instead of LP."""
    n_taxa = len(x0)
    traj = np.full((n_steps + 1, n_taxa), np.nan)
    mu_traj = np.full((n_steps, n_taxa), np.nan)
    gr_traj = np.full((n_steps,), np.nan)
    traj[0] = x0.copy()
    x = x0.copy()

    last_mu, last_gr = np.zeros(n_taxa), 0.0
    for step in range(n_steps):
        if np.max(x) >= convergence_threshold:
            for rem in range(step, n_steps):
                traj[rem + 1] = x
                mu_traj[rem] = last_mu
                gr_traj[rem] = last_gr
            return traj, mu_traj, gr_traj

        mu = surrogate_predict(model_nn, x_scaler, y_scaler, x[None, :])[0]
        comm_gr = float(x @ mu)
        last_mu, last_gr = mu.copy(), comm_gr

        mu_traj[step] = mu
        gr_traj[step] = comm_gr
        x = euler_step(x, mu, dt)
        traj[step + 1] = x

    return traj, mu_traj, gr_traj


def plot_single(global_idx, t_axis, taxa_ids, lp_traj, sur_traj, plots_dir):
    n_taxa = len(taxa_ids)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1.0, max(n_taxa, 2)))
    tax_color = dict(zip(taxa_ids, colors))

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for j, taxon in enumerate(taxa_ids):
        color = tax_color[taxon]
        ax.plot(t_axis, lp_traj[:, j], color=color, linewidth=2.0, linestyle="-")
        ax.plot(t_axis, sur_traj[:, j], color=color, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative abundance")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_title(f"config_{global_idx:04d}  (— LP, - - Surrogate)", fontsize=10)

    path = plots_dir / f"convergence_{global_idx:04d}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading model")
    t0 = time.time()
    comm = load_pickle(str(args.model_path))
    print(f"  Loaded in {time.time()-t0:.1f} s")
    taxa_ids = list(comm.taxa)
    n_taxa = len(taxa_ids)
    default_medium = dict(comm.medium)
    apply_medium_and_unlock(comm, default_medium)
    print(f"  Taxa ({n_taxa}): {taxa_ids}")

    section("2. Loading surrogate checkpoint")
    model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    print(f"  Checkpoint: {args.checkpoint.name}")

    section(f"3. Loading ICs from {args.ics_file}")
    ics = np.load(args.ics_file, allow_pickle=True)
    ics_taxa_ids = [str(t) for t in ics["taxa_ids"]]
    if ics_taxa_ids != taxa_ids:
        raise SystemExit(
            f"taxa_ids mismatch: ICs file has {ics_taxa_ids}, model has {taxa_ids}. "
            "Refusing to run a different community build/taxon ordering."
        )
    X_ics = ics["X"]
    n_total = len(X_ics)

    if args.n_shards is not None:
        if args.shard_index is None:
            raise SystemExit("--n-shards requires --shard-index")
        global_indices = np.array_split(np.arange(n_total), args.n_shards)[args.shard_index]
    else:
        global_indices = np.arange(n_total)
    print(f"  Total ICs: {n_total}   This shard: {len(global_indices)} "
          f"(indices {global_indices[0]}..{global_indices[-1]})")

    section(f"4. Running convergence-truncated trajectories "
            f"(n_steps<={args.n_steps}, dt={args.dt}, threshold={args.convergence_threshold})")
    t_axis = np.arange(args.n_steps + 1) * args.dt

    lp_trajs, lp_mu_trajs, lp_gr_trajs = [], [], []
    sur_trajs, sur_mu_trajs, sur_gr_trajs = [], [], []
    x0_list, converged_steps, plot_paths = [], [], []
    flat_X, flat_Y = [], []

    t_shard = time.time()
    for local_i, global_idx in enumerate(global_indices):
        x0 = X_ics[global_idx]
        lp_traj, lp_mu, lp_gr, n_lp_calls = run_lp_trajectory(
            comm, x0, taxa_ids, default_medium, args.fraction,
            args.n_steps, args.dt, args.convergence_threshold,
        )
        sur_traj, sur_mu, sur_gr = run_surrogate_trajectory(
            model_nn, x_scaler, y_scaler, x0,
            args.n_steps, args.dt, args.convergence_threshold,
        )

        # Only the first n_lp_calls rows of lp_traj/lp_mu are real, distinct LP
        # solves -- everything from n_lp_calls onward is the convergence-tail
        # fill (traj/mu repeated at the converged state, not NaN), so slicing
        # by n_lp_calls (not an isnan mask) avoids duplicating the same point
        # many times over.
        flat_X.append(lp_traj[:n_lp_calls])
        flat_Y.append(lp_mu[:n_lp_calls])

        path = plot_single(global_idx, t_axis, taxa_ids, lp_traj, sur_traj, args.plots_dir)

        lp_trajs.append(lp_traj); lp_mu_trajs.append(lp_mu); lp_gr_trajs.append(lp_gr)
        sur_trajs.append(sur_traj); sur_mu_trajs.append(sur_mu); sur_gr_trajs.append(sur_gr)
        x0_list.append(x0); converged_steps.append(n_lp_calls); plot_paths.append(str(path))

        if (local_i + 1) % 10 == 0 or local_i == 0 or local_i == len(global_indices) - 1:
            dominant = taxa_ids[int(np.argmax(lp_traj[~np.isnan(lp_traj).any(axis=1)][-1]))]
            print(f"  [{local_i+1:>4}/{len(global_indices)}]  config={global_idx:>4}  "
                  f"lp_calls={n_lp_calls:>3}  dominant={dominant}  -> {Path(path).name}")

    shard_elapsed = time.time() - t_shard
    total_lp_calls = int(np.sum(converged_steps))
    worst_case_calls = len(global_indices) * args.n_steps
    print(f"\n  Shard done in {shard_elapsed:.0f} s ({shard_elapsed/3600:.2f} h)")
    print(f"  LP calls used: {total_lp_calls}  (vs. {worst_case_calls} if none had "
          f"converged early -- {100*total_lp_calls/worst_case_calls:.1f}%)")
    print(f"  Convergence steps: mean={np.mean(converged_steps):.1f}  "
          f"median={np.median(converged_steps):.0f}  max={np.max(converged_steps)}")

    section("5. Saving")
    np.savez_compressed(
        args.output,
        taxa_ids=taxa_ids,
        global_indices=global_indices,
        x0=np.stack(x0_list),
        lp_trajectories=np.stack(lp_trajs),
        lp_mu_trajs=np.stack(lp_mu_trajs),
        lp_gr_trajs=np.stack(lp_gr_trajs),
        surrogate_trajectories=np.stack(sur_trajs),
        surrogate_mu_trajs=np.stack(sur_mu_trajs),
        surrogate_gr_trajs=np.stack(sur_gr_trajs),
        n_lp_calls=np.array(converged_steps),
        plot_paths=np.array(plot_paths),
        # Flattened (abundance, growth_rate) pairs actually visited along the LP
        # trajectories in this shard -- same schema as hvsc1_generate_data.py's
        # X/Y, usable as extra ground-truth data IF wanted (not auto-merged;
        # these are correlated along a trajectory, not i.i.d. samples).
        X=np.vstack(flat_X) if flat_X else np.zeros((0, n_taxa)),
        Y=np.vstack(flat_Y) if flat_Y else np.zeros((0, n_taxa)),
        dt=args.dt, n_steps=args.n_steps, fraction=args.fraction,
        convergence_threshold=args.convergence_threshold,
    )
    print(f"  Saved: {args.output}")
    print(f"  Plots: {args.plots_dir} ({len(plot_paths)} files)")


if __name__ == "__main__":
    main()
