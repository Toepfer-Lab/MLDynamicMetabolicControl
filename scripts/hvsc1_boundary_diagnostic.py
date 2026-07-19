"""
Decision-boundary diagnostic for the hvsc1 surrogate: checks whether a
trajectory-level LP-vs-surrogate mismatch (e.g. the "original" profile) is
explained by the true input sitting on/near a sharp decision boundary
between competing taxa, rather than a broad extrapolation failure.

Two complementary views, both using small Dirichlet(x*kappa) jitter around
a reference point (concentrated near x for large kappa, respects the
simplex natively -- no ad hoc clip-and-renormalize needed):

  A) Perturbed-starting-point robustness: run --n-perturb-starts surrogate
     trajectories from jittered variants of the reference profile's
     starting point, overlaid against the single LP reference trajectory.
     If the perturbed surrogate trajectories fan out to different final
     dominant taxa, the surrogate's basin of attraction is locally unstable
     right at this starting point -- a real boundary signature. If they all
     agree tightly with each other (even while disagreeing with LP), the
     mismatch is a systematic bias, not a boundary artifact.

  B) Local neighborhood sensitivity along the true (LP) trajectory: at each
     LP time step's true x[t] (oracle-style, not compounding), sample
     --n-neighbors jittered neighbors and evaluate the surrogate on all of
     them. Reports the predicted growth-rate gap between --taxon-a (LP's
     answer) and --taxon-b (the surrogate's wrong answer) across the
     neighborhood at each step, next to the unperturbed point prediction's
     own gap. Steps where the neighborhood band straddles zero (some
     neighbors favor a, some favor b) are decision-boundary steps; steps
     where the whole neighborhood agrees (band doesn't cross zero) are a
     confident, systematic error instead.

Usage:
    python scripts/hvsc1_boundary_diagnostic.py --profile original \\
        --taxon-a 1234 --taxon-b 946 --output-dir plots/hvsc1_al_1
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "trained_models"
                           / "hvsc1_community_input-27_output-27_hidden-128.pt")
    p.add_argument("--lp-trajectories", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_trajectories.npz")
    p.add_argument("--profile", type=str, default="original")
    p.add_argument("--taxon-a", type=str, default=None,
                   help="First taxon of the contested pair (default: LP's final dominant)")
    p.add_argument("--taxon-b", type=str, default=None,
                   help="Second taxon of the contested pair (default: surrogate's "
                        "predicted dominant at t=0, if different from --taxon-a)")
    p.add_argument("--n-perturb-starts", type=int, default=6,
                   help="K for view A: number of jittered starting points")
    p.add_argument("--n-neighbors", type=int, default=50,
                   help="M for view B: number of jittered neighbors per timestep")
    p.add_argument("--kappa", type=float, default=1000.0,
                   help="Dirichlet(x*kappa) concentration -- larger kappa = tighter jitter "
                        "around x. ~1000 gives roughly 1-2%% absolute perturbation at "
                        "typical mid-simplex abundance scales.")
    p.add_argument("--convergence-threshold", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=11_000,
                   help="Fresh seed block, distinct from all other seed ranges in use.")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "plots" / "hvsc1_al_1")
    return p.parse_args()


def euler_step(x, mu, dt):
    x_new = x * (1.0 + mu * dt)
    x_new = np.maximum(x_new, MIN_ABUND)
    x_new /= x_new.sum()
    return x_new


def run_surrogate_trajectory(model_nn, x_scaler, y_scaler, x0, n_steps, dt, convergence_threshold):
    n_taxa = len(x0)
    traj = np.full((n_steps + 1, n_taxa), np.nan)
    traj[0] = x0.copy()
    x = x0.copy()
    last_mu = np.zeros(n_taxa)
    for step in range(n_steps):
        if np.max(x) >= convergence_threshold:
            for rem in range(step, n_steps):
                traj[rem + 1] = x
            return traj
        mu = surrogate_predict(model_nn, x_scaler, y_scaler, x[None, :])[0]
        last_mu = mu
        x = euler_step(x, mu, dt)
        traj[step + 1] = x
    return traj


def part_a_perturbed_starts(args, model_nn, x_scaler, y_scaler, taxa_ids, x0, lp_traj, dt, n_steps):
    section(f"A. Perturbed-starting-point robustness ({args.n_perturb_starts} runs, "
            f"kappa={args.kappa})")
    rng = np.random.default_rng(args.seed)
    n_taxa = len(taxa_ids)
    perturbed_trajs = []
    final_doms = []
    for k in range(args.n_perturb_starts):
        x0_k = rng.dirichlet(x0 * args.kappa)
        traj_k = run_surrogate_trajectory(model_nn, x_scaler, y_scaler, x0_k,
                                          n_steps, dt, args.convergence_threshold)
        perturbed_trajs.append(traj_k)
        final_x = traj_k[~np.isnan(traj_k).any(axis=1)][-1]
        dom = taxa_ids[int(np.argmax(final_x))]
        final_doms.append(dom)
        print(f"  perturbed run {k}: final dominant = {dom}")

    lp_dom = taxa_ids[int(np.argmax(lp_traj[-1]))]
    n_agree_lp = sum(d == lp_dom for d in final_doms)
    print(f"\n  LP final dominant: {lp_dom}")
    print(f"  Perturbed surrogate runs agreeing with LP: {n_agree_lp}/{args.n_perturb_starts}")
    unique_doms = set(final_doms)
    if len(unique_doms) == 1:
        print(f"  All {args.n_perturb_starts} perturbed runs agree with EACH OTHER "
              f"(all -> {unique_doms.pop()}) -- looks like a systematic bias, not a "
              f"local decision-boundary artifact.")
    else:
        print(f"  Perturbed runs disagree among themselves ({unique_doms}) -- consistent "
              f"with a locally unstable decision boundary right at this starting point.")

    colors = plt.get_cmap("tab20")(np.linspace(0, 1.0, max(n_taxa, 2)))
    tax_color = dict(zip(taxa_ids, colors))
    t_axis = np.arange(n_steps + 1) * dt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for traj_k in perturbed_trajs:
        for j, taxon in enumerate(taxa_ids):
            ax.plot(t_axis, traj_k[:, j], color=tax_color[taxon],
                    linewidth=1.0, linestyle="--", alpha=0.45)
    for j, taxon in enumerate(taxa_ids):
        ax.plot(t_axis, lp_traj[:, j], color=tax_color[taxon], linewidth=2.2, linestyle="-")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative abundance")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_title(f"{args.profile}: LP (—, bold) vs {args.n_perturb_starts} jittered "
                 f"surrogate starts (- -, faint)\nkappa={args.kappa}", fontsize=10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"boundary_diagnostic_{args.profile}_perturbed_starts.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {path}")


def part_b_neighborhood_sensitivity(args, model_nn, x_scaler, y_scaler, taxa_ids,
                                    lp_traj, dt, n_steps, idx_a, idx_b):
    section(f"B. Local neighborhood sensitivity along the LP trajectory "
            f"({args.n_neighbors} neighbors/step, kappa={args.kappa})")
    rng = np.random.default_rng(args.seed + 1)

    t_steps = np.arange(n_steps + 1)
    point_gap = np.full(len(t_steps), np.nan)
    neighbor_mean = np.full(len(t_steps), np.nan)
    neighbor_min = np.full(len(t_steps), np.nan)
    neighbor_max = np.full(len(t_steps), np.nan)
    neighbor_std = np.full(len(t_steps), np.nan)

    for i, t in enumerate(t_steps):
        x_t = lp_traj[t]
        if np.isnan(x_t).any():
            continue
        mu_point = surrogate_predict(model_nn, x_scaler, y_scaler, x_t[None, :])[0]
        point_gap[i] = mu_point[idx_a] - mu_point[idx_b]

        neighbors = rng.dirichlet(x_t * args.kappa, size=args.n_neighbors)
        mu_neighbors = surrogate_predict(model_nn, x_scaler, y_scaler, neighbors)
        gaps = mu_neighbors[:, idx_a] - mu_neighbors[:, idx_b]
        neighbor_mean[i] = gaps.mean()
        neighbor_min[i] = gaps.min()
        neighbor_max[i] = gaps.max()
        neighbor_std[i] = gaps.std()

    valid = ~np.isnan(point_gap)
    n_boundary_steps = int(np.sum((neighbor_min[valid] < 0) & (neighbor_max[valid] > 0)))
    n_confident_wrong = int(np.sum((point_gap[valid] < 0) & (neighbor_max[valid] < 0)))
    n_confident_right = int(np.sum((point_gap[valid] > 0) & (neighbor_min[valid] > 0)))
    print(f"  Valid steps: {valid.sum()}")
    print(f"  Steps where the neighborhood band straddles zero (boundary-like): {n_boundary_steps}")
    print(f"  Steps confidently wrong (point & whole neighborhood favor {taxa_ids[idx_b]}): {n_confident_wrong}")
    print(f"  Steps confidently right (point & whole neighborhood favor {taxa_ids[idx_a]}): {n_confident_right}")

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    t_axis = t_steps * dt
    ax.fill_between(t_axis[valid], neighbor_min[valid], neighbor_max[valid],
                    color="tab:blue", alpha=0.25, label=f"neighbor min-max (n={args.n_neighbors})")
    ax.plot(t_axis[valid], neighbor_mean[valid], color="tab:blue", linewidth=1.5,
           label="neighbor mean")
    ax.plot(t_axis[valid], point_gap[valid], color="black", linewidth=1.8,
           linestyle="--", label="unperturbed point prediction")
    ax.axhline(0, color="red", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(f"Predicted growth-rate gap: {taxa_ids[idx_a]} − {taxa_ids[idx_b]}")
    ax.set_title(f"{args.profile}: local sensitivity of {taxa_ids[idx_a]} vs {taxa_ids[idx_b]} "
                 f"ranking\n(above zero line = favors {taxa_ids[idx_a]} [LP's answer], "
                 f"below = favors {taxa_ids[idx_b]})", fontsize=10)
    ax.legend(fontsize=8)

    path = args.output_dir / f"boundary_diagnostic_{args.profile}_neighborhood_sensitivity.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    args = parse_args()

    section(f"1. Loading LP trajectories from {args.lp_trajectories}")
    lp = np.load(args.lp_trajectories, allow_pickle=True)
    taxa_ids = [str(t) for t in lp["taxa_ids"]]
    profile_names = [str(p) for p in lp["profile_names"]]
    if args.profile not in profile_names:
        raise SystemExit(f"Profile {args.profile!r} not found in {profile_names}")
    p_idx = profile_names.index(args.profile)
    lp_traj = lp["trajectories"][p_idx]
    dt = float(lp["dt"])
    n_steps = int(lp["n_steps"])
    x0 = lp_traj[0]
    print(f"  Profile: {args.profile}   dt={dt}  n_steps={n_steps}")

    section("2. Loading surrogate checkpoint")
    model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    print(f"  Checkpoint: {args.checkpoint.name}")

    lp_final_dom = taxa_ids[int(np.argmax(lp_traj[-1]))]
    mu0 = surrogate_predict(model_nn, x_scaler, y_scaler, x0[None, :])[0]
    sur_t0_dom = taxa_ids[int(np.argmax(mu0))]
    taxon_a = args.taxon_a or lp_final_dom
    taxon_b = args.taxon_b or (sur_t0_dom if sur_t0_dom != taxon_a
                               else taxa_ids[int(np.argsort(mu0)[::-1][1])])
    idx_a, idx_b = taxa_ids.index(taxon_a), taxa_ids.index(taxon_b)
    print(f"  Contested pair: taxon_a={taxon_a} (LP final dominant), "
          f"taxon_b={taxon_b} (surrogate's t=0 pick)")

    part_a_perturbed_starts(args, model_nn, x_scaler, y_scaler, taxa_ids, x0, lp_traj, dt, n_steps)
    part_b_neighborhood_sensitivity(args, model_nn, x_scaler, y_scaler, taxa_ids,
                                    lp_traj, dt, n_steps, idx_a, idx_b)


if __name__ == "__main__":
    main()
