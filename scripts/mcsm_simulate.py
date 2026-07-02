"""
Community abundance dynamics — landscape sweep and simulation loop.

Two tasks in one job:

  Task 1 — Landscape sweep (N_LANDSCAPE Dirichlet samples):
    For each sampled abundance vector, run cooperative_tradeoff and record
    per-taxon growth rates. Gives a picture of the growth-rate landscape
    across the full abundance simplex before committing to a surrogate design.

  Task 2 — Simulation loop from multiple starting points:
    At each step: set_abundance → cooperative_tradeoff → update abundances
    using the replicator / Euler step → renormalise → repeat.
    Shows whether dynamics converge, how fast, and flags any regimes where
    the LP becomes infeasible or drops community growth unexpectedly.

Normalisation convention:
    After each Euler step  x_new = x * (1 + mu * dt)  the vector is
    renormalised to sum=1. This is equivalent to the replicator equation
    dx_i/dt = x_i * (mu_i - mean_fitness) for small dt, and guarantees
    sum(x)=1 at every step without any extra bookkeeping.

Results saved to:
    results/mcsm_landscape.npz   — landscape sweep
    results/mcsm_trajectories.npz — simulation trajectories
"""

import time
import numpy as np
import pandas as pd
import cobra
import micom
from micom import load_pickle
from pathlib import Path

cobra.Configuration.solver = "cplex"

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
MODEL_PATH  = REPO_ROOT / "model" / "dcom.pickle"
MEDIUM_PATH = REPO_ROOT / "data"  / "Completed_maize_leaf_medium.csv"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── simulation parameters ─────────────────────────────────────────────────────
FRACTION     = 0.5     # cooperative tradeoff fraction
DT           = 0.1     # time step (hours)
N_STEPS      = 80      # steps per trajectory = DT*N_STEPS hours total
N_LANDSCAPE  = 200     # Dirichlet samples for the landscape sweep
MIN_ABUND    = 1e-8    # floor to prevent numerical extinction
SEED         = 42


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def apply_medium_and_unlock(comm, medium_dict):
    """Apply the fixed medium and unlock internal exchanges."""
    comm.medium = medium_dict
    for rxn in comm.internal_exchanges:
        rxn.bounds = (-1000.0, 1000.0)


def solve(comm, abundance_vec, taxa_ids, medium_dict):
    """
    Set abundance, re-apply medium, solve. Returns (growth_rate_vec, status,
    community_gr, elapsed_s). growth_rate_vec is aligned to taxa_ids order.
    """
    comm.set_abundance(pd.Series(dict(zip(taxa_ids, abundance_vec))))
    apply_medium_and_unlock(comm, medium_dict)
    t0 = time.time()
    sol = comm.cooperative_tradeoff(fraction=FRACTION, pfba=False, fluxes=False)
    elapsed = time.time() - t0
    # Align to taxa_ids, drop the 'medium' row via reindex
    mu = sol.members.reindex(taxa_ids)["growth_rate"].values.astype(float)
    return mu, sol.status, float(sol.growth_rate), elapsed


def euler_step(x, mu, dt):
    """
    One step of the replicator / renormalised Euler update.
    x and mu are aligned numpy arrays of length n_taxa.
    """
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)   # extinction floor
    x_new /= x_new.sum()                    # renormalise to sum=1
    return x_new


# ── 1. Load model and medium ──────────────────────────────────────────────────
section("1. Loading model and applying medium")
t0 = time.time()
comm = load_pickle(str(MODEL_PATH))
print(f"  Loaded in {time.time()-t0:.1f} s")

medium_df   = pd.read_csv(MEDIUM_PATH, index_col=0)
medium_dict = dict(zip(medium_df["reaction"], medium_df["flux"]))
apply_medium_and_unlock(comm, medium_dict)

taxa_ids = list(comm.taxa)
n_taxa   = len(taxa_ids)
print(f"  Taxa ({n_taxa}): {taxa_ids}")
print(f"  Medium components accepted: {len(comm.medium)}")


# ── 2. Landscape sweep ────────────────────────────────────────────────────────
section(f"2. Landscape sweep — {N_LANDSCAPE} Dirichlet samples")

rng     = np.random.default_rng(SEED)
samples = rng.dirichlet(np.ones(n_taxa), size=N_LANDSCAPE)

landscape_x  = np.zeros((N_LANDSCAPE, n_taxa))
landscape_mu = np.zeros((N_LANDSCAPE, n_taxa))
landscape_gr = np.zeros(N_LANDSCAPE)
landscape_ok = np.ones(N_LANDSCAPE, dtype=bool)

t_sweep_start = time.time()
for i, x in enumerate(samples):
    mu, status, comm_gr, elapsed = solve(comm, x, taxa_ids, medium_dict)
    landscape_x[i]  = x
    landscape_mu[i] = mu
    landscape_gr[i] = comm_gr
    if status != "optimal":
        landscape_ok[i] = False
    if (i + 1) % 20 == 0 or i == 0:
        print(f"  [{i+1:>3}/{N_LANDSCAPE}]  status={status}  "
              f"comm_gr={comm_gr:.4f}  time={elapsed:.1f}s")

sweep_time = time.time() - t_sweep_start
n_ok = landscape_ok.sum()
print(f"\n  Sweep done in {sweep_time:.0f} s  ({n_ok}/{N_LANDSCAPE} optimal)")
print(f"  Community growth rate: min={landscape_gr[landscape_ok].min():.4f}  "
      f"max={landscape_gr[landscape_ok].max():.4f}  "
      f"mean={landscape_gr[landscape_ok].mean():.4f}")
print(f"\n  Per-taxon growth rate range across all samples:")
for j, t in enumerate(taxa_ids):
    lo = landscape_mu[landscape_ok, j].min()
    hi = landscape_mu[landscape_ok, j].max()
    print(f"    {t:<12}  [{lo:.4f}, {hi:.4f}]  spread={hi-lo:.4f}")

np.savez_compressed(
    RESULTS_DIR / "mcsm_landscape.npz",
    taxa_ids=taxa_ids, x=landscape_x, mu=landscape_mu,
    community_gr=landscape_gr, optimal=landscape_ok,
    fraction=FRACTION, n_samples=N_LANDSCAPE, seed=SEED,
)
print(f"\n  Saved: {RESULTS_DIR / 'mcsm_landscape.npz'}")


# ── 3. Simulation loop ────────────────────────────────────────────────────────
section(f"3. Simulation — {N_STEPS} steps × dt={DT}h = {N_STEPS*DT:.1f}h total")

# Starting conditions: original abundance + 4 Dirichlet draws
orig_abund = (comm.taxonomy
              .set_index("id")["abundance"]
              .reindex(taxa_ids).values.astype(float))
orig_abund /= orig_abund.sum()

start_profiles = {
    "original"    : orig_abund,
    "uniform"     : np.ones(n_taxa) / n_taxa,
    "dirichlet_a" : rng.dirichlet(np.ones(n_taxa)),
    "dirichlet_b" : rng.dirichlet(np.ones(n_taxa)),
    "dirichlet_c" : rng.dirichlet(np.ones(n_taxa)),
}

all_trajectories = {}   # name → (n_steps+1, n_taxa)
all_mu_trajs     = {}   # name → (n_steps, n_taxa)
all_gr_trajs     = {}   # name → (n_steps,)
all_times        = {}   # name → wall time

print(f"\n  Initial abundances:")
header = f"  {'profile':<16}" + "".join(f"  {t:<10}" for t in taxa_ids)
print(header)
for name, x0 in start_profiles.items():
    row = f"  {name:<16}" + "".join(f"  {v:<10.4f}" for v in x0)
    print(row)

for name, x0 in start_profiles.items():
    print(f"\n  ── Trajectory: {name} ──")
    traj    = np.full((N_STEPS + 1, n_taxa), np.nan)
    mu_traj = np.full((N_STEPS,     n_taxa), np.nan)
    gr_traj = np.full((N_STEPS,),           np.nan)
    traj[0] = x0.copy()
    x = x0.copy()
    t_sim = time.time()

    for step in range(N_STEPS):
        mu, status, comm_gr, elapsed = solve(comm, x, taxa_ids, medium_dict)
        if status != "optimal":
            print(f"    step {step}: status={status}, stopping early.")
            break
        mu_traj[step] = mu
        gr_traj[step] = comm_gr
        x = euler_step(x, mu, DT)
        traj[step + 1] = x

        if (step + 1) % 10 == 0:
            dominant = taxa_ids[np.argmax(x)]
            print(f"    step {step+1:>3}  t={( step+1)*DT:5.1f}h  "
                  f"comm_gr={comm_gr:.4f}  dominant={dominant} ({x.max():.3f})")

    sim_elapsed = time.time() - t_sim
    all_trajectories[name] = traj
    all_mu_trajs[name]     = mu_traj
    all_gr_trajs[name]     = gr_traj
    all_times[name]        = sim_elapsed

    final_x = traj[~np.isnan(traj).any(axis=1)][-1]
    dominant = taxa_ids[np.argmax(final_x)]
    print(f"  Final state (t={N_STEPS*DT:.1f}h):  dominant={dominant} ({final_x.max():.4f})")
    print(f"  Wall time: {sim_elapsed:.0f} s")

# ── 4. Save trajectory results ────────────────────────────────────────────────
section("4. Saving trajectory results")
np.savez_compressed(
    RESULTS_DIR / "mcsm_trajectories.npz",
    taxa_ids      = taxa_ids,
    profile_names = list(all_trajectories.keys()),
    trajectories  = np.stack(list(all_trajectories.values())),  # (n_profiles, n_steps+1, n_taxa)
    mu_trajs      = np.stack(list(all_mu_trajs.values())),      # (n_profiles, n_steps, n_taxa)
    gr_trajs      = np.stack(list(all_gr_trajs.values())),      # (n_profiles, n_steps)
    dt=DT, n_steps=N_STEPS, fraction=FRACTION, seed=SEED,
)
print(f"  Saved: {RESULTS_DIR / 'mcsm_trajectories.npz'}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
section("5. Summary")
print(f"  Landscape: {N_LANDSCAPE} samples, {n_ok} optimal, {sweep_time:.0f}s")
for name, t_wall in all_times.items():
    final_x = all_trajectories[name]
    final_x = final_x[~np.isnan(final_x).any(axis=1)][-1]
    dominant = taxa_ids[np.argmax(final_x)]
    print(f"  {name:<16}  wall={t_wall:.0f}s  "
          f"final dominant={dominant} ({final_x.max():.3f})")
