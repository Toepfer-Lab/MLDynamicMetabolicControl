"""
Run the HvSC1 community abundance ODE using the trained surrogate in place of LP.

Two modes:

  Normal mode (default):
    Replicates the Euler simulation loop from hvsc1_simulate.py using the
    surrogate NN instead of cooperative_tradeoff.  Starting conditions are
    read from the LP trajectory file (identical initial states).
    Output: results/hvsc1_surrogate_trajectories.npz

  Oracle mode (--oracle):
    Evaluates the surrogate at the LP's *own* abundance vectors at every step,
    rather than following a compounding surrogate trajectory.  This separates
    pure prediction error from trajectory-accumulation error: if the MAE is low
    but dominant-taxon agreement is wrong even at early steps, the surrogate has
    a directional bias; if agreement is fine at early steps but breaks later, it
    is distribution shift (surrogate never saw near-boundary states in training).
    Output: results/hvsc1_oracle_compare.npz

Usage:
    # normal simulation
    python scripts/hvsc1_ode_surrogate.py

    # oracle comparison (requires LP trajectory file with mu_trajs)
    python scripts/hvsc1_ode_surrogate.py --oracle
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--lp-trajectories", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_trajectories.npz")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_surrogate_trajectories.npz")
    p.add_argument("--oracle-output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_oracle_compare.npz",
                   help="Output path for oracle comparison data (--oracle mode only)")
    p.add_argument("--n-steps", type=int,   default=80)
    p.add_argument("--dt",      type=float, default=0.1)
    p.add_argument("--oracle",  action="store_true",
                   help="Oracle mode: evaluate surrogate at LP trajectory points, "
                        "not on compounding surrogate trajectory")
    return p.parse_args()


def euler_step(x, mu, dt):
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def fmt_top3(mu, taxa_ids):
    """Return a short string showing the top-3 taxa by growth rate."""
    top3 = np.argsort(mu)[::-1][:3]
    parts = [f"{taxa_ids[i]}={mu[i]:.2f}" for i in top3]
    return "  ".join(parts)


def run_normal(args, model_nn, x_scaler, y_scaler, lp):
    """Compounding surrogate simulation — surrogate drives its own trajectory."""
    taxa_ids      = list(lp["taxa_ids"])
    profile_names = list(lp["profile_names"])
    lp_trajs      = lp["trajectories"]   # (n_profiles, n_steps+1, n_taxa)
    n_taxa        = len(taxa_ids)
    n_profiles    = len(profile_names)

    section("3. Surrogate simulation loop (normal mode)")
    all_trajectories = []
    all_mu_trajs     = []
    all_comm_gr      = []

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
            gr_traj[step] = float(x @ mu)
            x = euler_step(x, mu, args.dt)
            traj[step + 1] = x

            if (step + 1) % 10 == 0:
                dominant = taxa_ids[np.argmax(x)]
                top3_str = fmt_top3(mu, taxa_ids)
                print(f"    step {step+1:>3}  t={(step+1)*args.dt:5.1f}h  "
                      f"dominant={dominant} ({x.max():.3f})  "
                      f"top-3 mu: {top3_str}")

        final_x  = traj[-1]
        dominant = taxa_ids[np.argmax(final_x)]
        print(f"  Final state (t={args.n_steps*args.dt:.1f}h): "
              f"dominant={dominant} ({final_x.max():.4f})")

        all_trajectories.append(traj)
        all_mu_trajs.append(mu_traj)
        all_comm_gr.append(gr_traj)

    print(f"\n  Total wall time: {time.time()-t_total:.1f} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        taxa_ids      = taxa_ids,
        profile_names = profile_names,
        trajectories  = np.stack(all_trajectories),
        mu_trajs      = np.stack(all_mu_trajs),
        gr_trajs      = np.stack(all_comm_gr),
        dt            = args.dt,
        n_steps       = args.n_steps,
        checkpoint    = str(args.checkpoint),
    )
    print(f"  Saved: {args.output}")


def run_oracle(args, model_nn, x_scaler, y_scaler, lp):
    """
    Oracle mode: evaluate surrogate at LP's actual x[t] at every step.

    This isolates prediction error from trajectory compounding.  The surrogate
    is asked "given the LP's current state, what growth rates do you predict?"
    and the answer is compared to what the LP actually produced.
    """
    taxa_ids      = list(lp["taxa_ids"])
    profile_names = list(lp["profile_names"])
    lp_trajs      = lp["trajectories"]   # (n_profiles, n_steps+1, n_taxa)
    lp_mu_trajs   = lp["mu_trajs"]       # (n_profiles, n_steps, n_taxa)
    n_taxa        = len(taxa_ids)
    n_profiles    = len(profile_names)

    section("3. Oracle comparison (surrogate evaluated at LP trajectory points)")
    print("  For each step: surrogate is given the LP's x[t], not its own x[t].")
    print("  This separates prediction error from trajectory-compounding error.\n")

    all_mu_lp         = []   # (n_profiles, n_steps, n_taxa)
    all_mu_sur_oracle = []   # same
    all_mae           = []   # (n_profiles, n_steps)
    all_dom_agree     = []   # (n_profiles, n_steps) bool

    t_total = time.time()
    for p, name in enumerate(profile_names):
        mu_lp_p    = lp_mu_trajs[p]            # (n_steps, n_taxa) — LP growth rates
        x_lp_p     = lp_trajs[p]               # (n_steps+1, n_taxa) — LP abundances
        n_steps_lp = (~np.isnan(mu_lp_p).any(axis=1)).sum()  # steps with valid LP mu

        mu_sur_p  = np.full_like(mu_lp_p, np.nan)
        mae_p     = np.full(mu_lp_p.shape[0], np.nan)
        dom_agree = np.zeros(mu_lp_p.shape[0], dtype=bool)

        print(f"\n  ── {name} ({n_steps_lp} valid LP steps) ──")
        n_agree = 0
        first_disagree = None

        for step in range(n_steps_lp):
            x_lp_t   = x_lp_p[step]           # LP abundance at step t
            mu_lp_t  = mu_lp_p[step]           # LP growth rates at step t
            mu_sur_t = surrogate_predict(
                model_nn, x_scaler, y_scaler, x_lp_t
            ).flatten()

            mu_sur_p[step]  = mu_sur_t
            mae_p[step]     = np.mean(np.abs(mu_lp_t - mu_sur_t))
            lp_dom  = np.argmax(mu_lp_t)
            sur_dom = np.argmax(mu_sur_t)
            agree   = (lp_dom == sur_dom)
            dom_agree[step] = agree

            if agree:
                n_agree += 1
            elif first_disagree is None:
                first_disagree = step

            if step == 0 or (step + 1) % 5 == 0 or not agree:
                tick = "✓" if agree else "✗"
                print(f"    step {step+1:>3}  {tick}  "
                      f"LP  dom={taxa_ids[lp_dom]:<6} ({mu_lp_t[lp_dom]:.2f})  "
                      f"Sur dom={taxa_ids[sur_dom]:<6} ({mu_sur_t[sur_dom]:.2f})  "
                      f"MAE={mae_p[step]:.3f}")

        agree_rate = n_agree / max(n_steps_lp, 1)
        print(f"\n  Agreement rate: {n_agree}/{n_steps_lp} = {agree_rate:.1%}")
        if first_disagree is not None:
            print(f"  First disagreement at step {first_disagree + 1}  "
                  f"(LP x at that point: dominant={taxa_ids[np.argmax(x_lp_p[first_disagree])]} "
                  f"{x_lp_p[first_disagree].max():.3f})")
        else:
            print("  Perfect dominant-taxon agreement throughout!")

        all_mu_lp.append(mu_lp_p)
        all_mu_sur_oracle.append(mu_sur_p)
        all_mae.append(mae_p)
        all_dom_agree.append(dom_agree)

    print(f"\n  Total oracle eval time: {time.time()-t_total:.2f} s")

    args.oracle_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.oracle_output,
        taxa_ids        = taxa_ids,
        profile_names   = profile_names,
        mu_lp           = np.stack(all_mu_lp),           # (n_profiles, n_steps, n_taxa)
        mu_sur_oracle   = np.stack(all_mu_sur_oracle),   # same
        mae_per_step    = np.stack(all_mae),              # (n_profiles, n_steps)
        dom_agree       = np.stack(all_dom_agree),        # (n_profiles, n_steps)
        dt              = args.dt,
        n_steps         = args.n_steps,
        checkpoint      = str(args.checkpoint),
    )
    print(f"\n  Oracle comparison saved: {args.oracle_output}")


def main():
    args = parse_args()

    section("1. Loading surrogate checkpoint")
    model_nn, x_scaler, y_scaler, metadata = load_surrogate_checkpoint(
        args.checkpoint, SurrogateNN
    )
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Metadata   : vman={metadata.get('vman')}, "
          f"condition={metadata.get('condition')}, "
          f"hidden_dim={metadata.get('hidden_dim')}, "
          f"n_layers={metadata.get('n_layers', 1)}")

    section("2. Loading LP trajectory data")
    lp = np.load(args.lp_trajectories, allow_pickle=True)
    taxa_ids      = list(lp["taxa_ids"])
    profile_names = list(lp["profile_names"])
    n_taxa        = len(taxa_ids)
    n_profiles    = len(profile_names)
    print(f"  Taxa ({n_taxa}): {taxa_ids[:5]}{'...' if n_taxa > 5 else ''}")
    print(f"  Profiles   : {profile_names}")

    if args.oracle:
        if "mu_trajs" not in lp:
            raise ValueError("LP trajectory file missing 'mu_trajs' key — "
                             "re-run hvsc1_simulate.py first")
        print(f"  Mode       : ORACLE (surrogate evaluated at LP's x[t])")
        run_oracle(args, model_nn, x_scaler, y_scaler, lp)
    else:
        print(f"  Mode       : NORMAL (surrogate drives own trajectory)")
        print(f"  Simulating {n_profiles} trajectories × {args.n_steps} steps × dt={args.dt}h")
        run_normal(args, model_nn, x_scaler, y_scaler, lp)


if __name__ == "__main__":
    main()
