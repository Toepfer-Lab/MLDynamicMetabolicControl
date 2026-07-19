"""
Convergence-time + stability-switch landscape sweep for hvsc1 (Use-case B).

Pure surrogate, no CPLEX. For each of N starting compositions (a large,
corner-aware mixture-sampled pool), runs a convergence-truncated Euler
trajectory (same early-stop rule as hvsc1_simulate.py/hvsc1_convergence_sweep.py:
stop once max(x) >= convergence_threshold) and also K Dirichlet-jittered
replicates of the SAME starting point, recording whether the replicates agree
with the unperturbed trajectory on the final dominant taxon.

Everything is vectorized across the whole (1+K)*N-row batch at once (masking
out rows that have already converged each step) rather than looping per
starting point in Python -- this is what makes N in the tens/hundreds of
thousands tractable in well under a minute.

Two per-IC outputs:
  convergence_time      -- steps until max(x) >= convergence_threshold (or the
                            step budget, if it never converges)
  agreement_fraction    -- fraction of the K jittered replicates whose final
                            dominant taxon matches the unperturbed trajectory's
                            own final dominant taxon (a local instability/
                            "switch" signal -- low agreement means this
                            starting point sits near a decision boundary)

Usage:
    python scripts/hvsc1_landscape_sweep.py --ics-file results/hvsc1_usecase_b_ics.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN                                     # noqa: E402

MIN_ABUND = 1e-8


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ics-file", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b_ics.npz")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--convergence-threshold", type=float, default=0.99)
    p.add_argument("--k-replicates", type=int, default=10)
    p.add_argument("--kappa", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=14_500,
                   help="Fresh sub-block, distinct from the ICs' own seed (14000)")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b" / "landscape_sweep.npz")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    section("1. Loading IC pool and surrogate checkpoint")
    ics = np.load(args.ics_file, allow_pickle=True)
    X0 = ics["X"].astype(np.float64)
    taxa_ids = [str(t) for t in ics["taxa_ids"]]
    N, n_taxa = X0.shape
    print(f"  ICs: {N}   taxa: {n_taxa}")

    model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    print(f"  Checkpoint: {args.checkpoint.name}")

    section(f"2. Building batch (1 unperturbed + {args.k_replicates} jittered replicates per IC)")
    rng = np.random.default_rng(args.seed)
    K = args.k_replicates
    alpha = X0 * args.kappa
    blocks = [X0]
    for _ in range(K):
        # Vectorized Dirichlet(alpha) via the Gamma construction (Y_i ~ Gamma(alpha_i),
        # X = Y/sum(Y)) -- numpy's rng.dirichlet doesn't support a different alpha per
        # row in one call, but rng.gamma does support an elementwise shape array, which
        # is exactly what's needed to jitter N different points in one vectorized call.
        g = rng.gamma(alpha)
        blocks.append(g / g.sum(axis=1, keepdims=True))
    x_batch = np.vstack(blocks)
    n_rows = x_batch.shape[0]
    print(f"  Batch shape: {x_batch.shape}  ({N} unperturbed + {K}x{N} replicates)")

    section(f"3. Vectorized convergence-truncated Euler loop (budget {args.n_steps} steps)")
    t0 = time.time()
    x = x_batch.copy()
    active = np.ones(n_rows, dtype=bool)
    conv_time = np.full(n_rows, args.n_steps, dtype=int)

    for step in range(args.n_steps):
        idx = np.where(active)[0]
        if idx.size == 0:
            print(f"  All rows converged by step {step}")
            break
        mu = surrogate_predict(model_nn, x_scaler, y_scaler, x[idx])
        x_new = x[idx] * (1.0 + mu * args.dt)
        x_new = np.maximum(x_new, MIN_ABUND)
        x_new /= x_new.sum(axis=1, keepdims=True)
        x[idx] = x_new
        newly_converged = idx[np.max(x_new, axis=1) >= args.convergence_threshold]
        conv_time[newly_converged] = step + 1
        active[newly_converged] = False
        if (step + 1) % 10 == 0:
            print(f"    step {step+1:>3}: {active.sum()} rows still active "
                  f"(elapsed {time.time()-t0:.1f}s)")

    sweep_wall_time_seconds = time.time() - t0
    final_dominant = np.argmax(x, axis=1)
    print(f"\n  Sweep wall time: {sweep_wall_time_seconds:.2f} s "
          f"({n_rows} trajectories, {args.n_steps} step budget)")

    section("4. Unpacking per-IC results")
    convergence_time = conv_time[:N]
    final_dominant_idx = final_dominant[:N]
    replicate_conv_time = conv_time[N:].reshape(K, N).T
    replicate_final_dominant = final_dominant[N:].reshape(K, N).T
    agreement_fraction = (replicate_final_dominant == final_dominant_idx[:, None]).mean(axis=1)

    print(f"  convergence_time: mean={convergence_time.mean():.2f}  "
          f"median={np.median(convergence_time):.0f}  max={convergence_time.max()}")
    print(f"  agreement_fraction: mean={agreement_fraction.mean():.4f}  "
          f"min={agreement_fraction.min():.4f}  "
          f"frac fully agreeing (==1.0): {(agreement_fraction == 1.0).mean():.4f}")

    section("5. Saving")
    np.savez_compressed(
        args.output,
        taxa_ids=taxa_ids,
        global_indices=np.arange(N),
        x0=X0,
        convergence_time=convergence_time,
        final_dominant_idx=final_dominant_idx,
        agreement_fraction=agreement_fraction,
        replicate_convergence_time=replicate_conv_time,
        replicate_final_dominant_idx=replicate_final_dominant,
        k_replicates=K, kappa=args.kappa,
        n_steps=args.n_steps, dt=args.dt, convergence_threshold=args.convergence_threshold,
        seed=args.seed, ics_file=str(args.ics_file), checkpoint=str(args.checkpoint),
        sweep_wall_time_seconds=sweep_wall_time_seconds,
    )
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
