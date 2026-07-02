"""
Benchmark: LP solver (cooperative_tradeoff) vs trained surrogate.

Runs N_LP trajectories with the LP solver and the same N_LP starting
conditions through the surrogate.  Additionally runs a larger surrogate-only
batch (N_SUR trajectories) to represent a realistic large-scale use case.

Results saved to results/mcsm_benchmark.npz.

The LP section requires CPLEX — run this script via sbatch.
"""

import sys
import time
from pathlib import Path

import cobra
import micom
import numpy as np
import pandas as pd
from micom import load_pickle

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from runtime_utils import load_surrogate_checkpoint, surrogate_predict  # noqa: E402
from surrogateNN import SurrogateNN                                     # noqa: E402

cobra.Configuration.solver = "cplex"

# ── parameters ────────────────────────────────────────────────────────────────
MODEL_PATH  = REPO_ROOT / "model" / "dcom.pickle"
MEDIUM_PATH = REPO_ROOT / "data"  / "Completed_maize_leaf_medium.csv"
CHECKPOINT  = REPO_ROOT / "trained_models" / "mcsm_community_input-6_output-6_hidden-64.pt"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_LP     = 5     # LP trajectories to run (representative sample)
N_SUR    = 100   # surrogate-only trajectories (realistic large-scale batch)
N_STEPS  = 80
DT       = 0.1
FRACTION = 0.5
MIN_ABUND = 1e-8
SEED     = 99    # different from training seed


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def apply_medium_and_unlock(comm, medium_dict):
    comm.medium = medium_dict
    for rxn in comm.internal_exchanges:
        rxn.bounds = (-1000.0, 1000.0)


def lp_solve(comm, x, taxa_ids, medium_dict):
    comm.set_abundance(pd.Series(dict(zip(taxa_ids, x))))
    apply_medium_and_unlock(comm, medium_dict)
    t0 = time.time()
    sol = comm.cooperative_tradeoff(fraction=FRACTION, pfba=False, fluxes=False)
    elapsed = time.time() - t0
    mu = sol.members.reindex(taxa_ids)["growth_rate"].values.astype(float)
    return mu, sol.status, elapsed


def surrogate_step(model_nn, x_scaler, y_scaler, x):
    return surrogate_predict(model_nn, x_scaler, y_scaler, x).flatten()


def euler_step(x, mu):
    x_new = x * (1.0 + mu * DT)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


# ── 1. Load model and medium ──────────────────────────────────────────────────
section("1. Loading LP model and surrogate checkpoint")
t0 = time.time()
comm = load_pickle(str(MODEL_PATH))
print(f"  Community model loaded in {time.time()-t0:.1f} s")

medium_df   = pd.read_csv(MEDIUM_PATH, index_col=0)
medium_dict = dict(zip(medium_df["reaction"], medium_df["flux"]))
apply_medium_and_unlock(comm, medium_dict)

taxa_ids = list(comm.taxa)
n_taxa   = len(taxa_ids)
print(f"  Taxa ({n_taxa}): {taxa_ids}")

model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(CHECKPOINT, SurrogateNN)
print(f"  Surrogate loaded: {CHECKPOINT.name}")

# ── 2. Generate benchmark starting conditions ─────────────────────────────────
section(f"2. Generating benchmark starting conditions (seed={SEED})")
rng        = np.random.default_rng(SEED)
starts_lp  = rng.dirichlet(np.ones(n_taxa), size=N_LP)    # LP starting points
starts_sur = rng.dirichlet(np.ones(n_taxa), size=N_SUR)   # larger surrogate batch

print(f"  LP benchmark:       {N_LP} trajectories × {N_STEPS} steps")
print(f"  Surrogate (matched): {N_LP} trajectories × {N_STEPS} steps")
print(f"  Surrogate (large):   {N_SUR} trajectories × {N_STEPS} steps")

# ── 3. LP benchmark ───────────────────────────────────────────────────────────
section(f"3. LP benchmark — {N_LP} trajectories")
lp_traj_times = np.zeros(N_LP)
lp_step_times = np.zeros((N_LP, N_STEPS))

for i, x0 in enumerate(starts_lp):
    print(f"\n  Trajectory {i+1}/{N_LP} ──")
    x = x0.copy()
    t_traj = time.time()
    for step in range(N_STEPS):
        mu, status, elapsed = lp_solve(comm, x, taxa_ids, medium_dict)
        lp_step_times[i, step] = elapsed
        if status != "optimal":
            print(f"    step {step}: non-optimal ({status}), aborting trajectory")
            break
        x = euler_step(x, mu)
        if (step + 1) % 20 == 0:
            dominant = taxa_ids[np.argmax(x)]
            print(f"    step {step+1:>3}  t={(step+1)*DT:5.1f}h  "
                  f"dominant={dominant} ({x.max():.3f})  step_time={elapsed:.2f}s")
    lp_traj_times[i] = time.time() - t_traj
    print(f"  Trajectory {i+1} done in {lp_traj_times[i]:.1f} s")

print(f"\n  LP summary:")
print(f"    mean time / trajectory : {lp_traj_times.mean():.1f} s")
print(f"    mean time / LP call    : {lp_step_times.mean():.3f} s")
print(f"    total                  : {lp_traj_times.sum():.1f} s")

# ── 4. Surrogate benchmark — matched starting points ─────────────────────────
section(f"4. Surrogate benchmark — {N_LP} matched trajectories")
sur_traj_times_matched = np.zeros(N_LP)

for i, x0 in enumerate(starts_lp):   # same starts as LP
    x = x0.copy()
    t_traj = time.time()
    for step in range(N_STEPS):
        mu = surrogate_step(model_nn, x_scaler, y_scaler, x)
        x  = euler_step(x, mu)
    sur_traj_times_matched[i] = time.time() - t_traj

print(f"  mean time / trajectory : {sur_traj_times_matched.mean()*1000:.3f} ms")
print(f"  total                  : {sur_traj_times_matched.sum()*1000:.1f} ms")

# ── 5. Surrogate benchmark — large batch ─────────────────────────────────────
section(f"5. Surrogate benchmark — {N_SUR} trajectories")
t_batch = time.time()
sur_traj_times_large = np.zeros(N_SUR)

for i, x0 in enumerate(starts_sur):
    x = x0.copy()
    t_traj = time.time()
    for step in range(N_STEPS):
        mu = surrogate_step(model_nn, x_scaler, y_scaler, x)
        x  = euler_step(x, mu)
    sur_traj_times_large[i] = time.time() - t_traj

sur_batch_total = time.time() - t_batch
print(f"  mean time / trajectory : {sur_traj_times_large.mean()*1000:.3f} ms")
print(f"  total wall time        : {sur_batch_total:.3f} s")

# ── 6. Summary and speedup ────────────────────────────────────────────────────
section("6. Speedup summary")
mean_lp_per_traj  = lp_traj_times.mean()
mean_sur_per_traj = sur_traj_times_matched.mean()
mean_lp_per_step  = lp_step_times.mean()

speedup_per_traj  = mean_lp_per_traj / mean_sur_per_traj
speedup_per_step  = mean_lp_per_step / (mean_sur_per_traj / N_STEPS)

extrap_lp_n_lp   = N_LP  * mean_lp_per_traj
extrap_lp_n_sur  = N_SUR * mean_lp_per_traj

print(f"  Mean LP time / trajectory   : {mean_lp_per_traj:.2f} s")
print(f"  Mean LP time / step         : {mean_lp_per_step:.3f} s")
print(f"  Mean surrogate / trajectory : {mean_sur_per_traj*1000:.3f} ms")
print(f"  Speedup per trajectory      : {speedup_per_traj:.0f}×")
print(f"  Speedup per step            : {speedup_per_step:.0f}×")
print(f"\n  Extrapolated LP time for {N_LP} trajectories  : {extrap_lp_n_lp:.0f} s "
      f"({extrap_lp_n_lp/3600:.2f} h)")
print(f"  Extrapolated LP time for {N_SUR} trajectories : {extrap_lp_n_sur:.0f} s "
      f"({extrap_lp_n_sur/3600:.2f} h)")
print(f"  Measured surrogate time for {N_SUR} trajectories: {sur_batch_total:.2f} s")

# ── 7. Save ───────────────────────────────────────────────────────────────────
section("7. Saving benchmark results")
out_path = RESULTS_DIR / "mcsm_benchmark.npz"
np.savez_compressed(
    out_path,
    # LP
    n_lp              = N_LP,
    lp_traj_times     = lp_traj_times,        # (N_LP,)  seconds per trajectory
    lp_step_times     = lp_step_times,        # (N_LP, N_STEPS) seconds per call
    # Surrogate matched
    sur_traj_times_matched = sur_traj_times_matched,  # (N_LP,)
    # Surrogate large batch
    n_sur             = N_SUR,
    sur_traj_times_large = sur_traj_times_large,      # (N_SUR,)
    sur_batch_total   = sur_batch_total,
    # simulation params
    n_steps = N_STEPS,
    dt      = DT,
    fraction = FRACTION,
    seed    = SEED,
    # taxa
    taxa_ids = taxa_ids,
)
print(f"  Saved: {out_path}")
