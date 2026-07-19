"""
LP-anchored surrogate sensitivity analysis for hvsc1.

Pick any point along a real LP trajectory (any profile, any time step -- not
necessarily t=0 or the final state) as an anchor. The anchor's own true
dynamics already come cheaply from LP directly (one trajectory is fast enough
on its own). What LP can't afford around that anchor is an exhaustive
per-taxon sensitivity sweep across all 27 dimensions -- that's what the
surrogate is for here.

Methodology (hand-rolled "per-taxon share-variance index", not Sobol/Saltelli):
the input is a 27-dim COMPOSITIONAL vector (always sums to 1), so standard
Sobol indices (which assume independent inputs) don't directly apply without
an isometric-log-ratio transform. Instead, for each taxon j: hold the other 26
taxa's RELATIVE proportions fixed at the anchor, deterministically sweep taxon
j's own share across its feasible range, and record the resulting output
curve. The variance of that curve is taxon j's sensitivity score; the full
curve is a free byproduct that also reveals exact "flip points" (the share at
which a competitor overtakes the anchor's original winner).

Usage:
    python scripts/hvsc1_sensitivity_anchor.py --profile original --time-step 45
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
from surrogateNN import SurrogateNN                                     # noqa: E402

DEGENERATE_OTHER_SUM = 1e-6


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
    p.add_argument("--time-step", type=int, default=0,
                   help="Index into trajectories[profile, t, :], valid 0..n_steps")
    p.add_argument("--grid-points", type=int, default=150, help="M, grid resolution per taxon")
    p.add_argument("--epsilon", type=float, default=1e-4,
                   help="Grid spans linspace(epsilon, 1-epsilon, M)")
    p.add_argument("--n-validation-points", type=int, default=40)
    p.add_argument("--seed", type=int, default=13_000,
                   help="Fresh seed block, distinct from all other seed ranges in use "
                        "(1000/2000/7000/8000/9000/10000/11000)")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "hvsc1_usecase_a")
    p.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "plots" / "hvsc1_usecase_a")
    p.add_argument("--top-n-curves", type=int, default=9,
                   help="How many top-scoring taxa to show in the response-curve small-multiples")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    section("1. Loading LP trajectory anchor")
    lp = np.load(args.lp_trajectories, allow_pickle=True)
    taxa_ids = [str(t) for t in lp["taxa_ids"]]
    profile_names = [str(p) for p in lp["profile_names"]]
    if args.profile not in profile_names:
        raise SystemExit(f"Profile {args.profile!r} not found in {profile_names}")
    p_idx = profile_names.index(args.profile)
    trajectories = lp["trajectories"]
    mu_trajs = lp["mu_trajs"]
    n_steps = int(lp["n_steps"])
    n_taxa = len(taxa_ids)

    if not (0 <= args.time_step <= n_steps):
        raise SystemExit(f"--time-step must be in [0, {n_steps}], got {args.time_step}")

    x = trajectories[p_idx, args.time_step, :].astype(float)

    # This file has no separate n_lp_calls/real-call marker (unlike
    # hvsc1_convergence_sweep.py's own output) -- detect a converged/backfilled
    # tail point heuristically instead of blocking on it.
    converged_like = bool(np.max(x) >= 0.99)
    repeated_mu = False
    if 0 < args.time_step < mu_trajs.shape[1]:
        repeated_mu = bool(np.allclose(mu_trajs[p_idx, args.time_step],
                                       mu_trajs[p_idx, args.time_step - 1]))
    if converged_like or repeated_mu:
        print("  WARNING: this anchor looks like a converged/back-filled tail point "
              "(repeated state) rather than a fresh dynamical one -- proceeding anyway "
              "since the anchor choice is the caller's, but flag this in any writeup.")

    print(f"  Profile: {args.profile}   time_step={args.time_step}/{n_steps}")
    print(f"  anchor x: max={x.max():.4f} at taxon {taxa_ids[int(np.argmax(x))]}")

    section("2. Loading surrogate checkpoint")
    model_nn, x_scaler, y_scaler, _ = load_surrogate_checkpoint(args.checkpoint, SurrogateNN)
    print(f"  Checkpoint: {args.checkpoint.name}")

    anchor_mu = surrogate_predict(model_nn, x_scaler, y_scaler, x[None, :])[0]
    anchor_winner_idx = int(np.argmax(anchor_mu))
    print(f"  anchor winner (surrogate): {taxa_ids[anchor_winner_idx]}  "
          f"mu={anchor_mu[anchor_winner_idx]:.3f}")

    section(f"3. Per-taxon deterministic sensitivity sweep (M={args.grid_points})")
    M = args.grid_points
    s_grid = np.linspace(args.epsilon, 1 - args.epsilon, M)

    sweep_X = np.full((n_taxa, M, n_taxa), np.nan)
    sweep_mu = np.full((n_taxa, M, n_taxa), np.nan)
    sweep_winner_idx = np.full((n_taxa, M), -1, dtype=int)
    sweep_gap = np.full((n_taxa, M), np.nan)
    anchor_winner_gr_curve = np.full((n_taxa, M), np.nan)
    skipped_taxa = []

    for j in range(n_taxa):
        x_other = x.copy()
        x_other[j] = 0.0
        other_sum = x_other.sum()
        if other_sum < DEGENERATE_OTHER_SUM:
            skipped_taxa.append(taxa_ids[j])
            print(f"  taxon {taxa_ids[j]:<6} SKIPPED (anchor's other-26 sum={other_sum:.2e}, "
                  f"degenerate -- anchor is already ~monoculture in this taxon)")
            continue
        X_sweep = (x_other[None, :] / other_sum) * (1.0 - s_grid)[:, None]
        X_sweep[:, j] = s_grid
        mu_sweep = surrogate_predict(model_nn, x_scaler, y_scaler, X_sweep)
        sweep_X[j] = X_sweep
        sweep_mu[j] = mu_sweep
        sweep_winner_idx[j] = np.argmax(mu_sweep, axis=1)
        sorted_mu = np.sort(mu_sweep, axis=1)
        sweep_gap[j] = sorted_mu[:, -1] - sorted_mu[:, -2]
        anchor_winner_gr_curve[j] = mu_sweep[:, anchor_winner_idx]

    section("4. Flip-point detection")
    flip_taxon_j, flip_grid_idx, flip_from_idx, flip_to_idx, flip_s_j = [], [], [], [], []
    flip_counts = {t: 0 for t in taxa_ids}
    for j in range(n_taxa):
        if taxa_ids[j] in skipped_taxa:
            continue
        w = sweep_winner_idx[j]
        changes = np.where(w[1:] != w[:-1])[0] + 1
        for m in changes:
            flip_taxon_j.append(taxa_ids[j])
            flip_grid_idx.append(int(m))
            flip_from_idx.append(int(w[m - 1]))
            flip_to_idx.append(int(w[m]))
            flip_s_j.append(float(s_grid[m]))
        flip_counts[taxa_ids[j]] = len(changes)
        print(f"  taxon {taxa_ids[j]:<6} {len(changes)} flip(s)")

    section("5. Sensitivity scores")
    score_winner_gr_var = np.full(n_taxa, np.nan)
    score_gap_var = np.full(n_taxa, np.nan)
    for j in range(n_taxa):
        if taxa_ids[j] in skipped_taxa:
            continue
        score_winner_gr_var[j] = np.nanvar(anchor_winner_gr_curve[j])
        score_gap_var[j] = np.nanvar(sweep_gap[j])

    ranked = sorted(range(n_taxa), key=lambda j: -np.nan_to_num(score_winner_gr_var[j], nan=-np.inf))
    print(f"  {'Taxon':<8}{'score_winner_gr_var':>22}{'score_gap_var':>16}{'n_flips':>10}")
    print(f"  {'-'*56}")
    for j in ranked:
        t = taxa_ids[j]
        if t in skipped_taxa:
            continue
        print(f"  {t:<8}{score_winner_gr_var[j]:>22.4f}{score_gap_var[j]:>16.4f}{flip_counts[t]:>10}")

    section("6. Saving results")
    out_path = args.output_dir / f"sensitivity_{args.profile}_t{args.time_step}.npz"
    np.savez_compressed(
        out_path,
        taxa_ids=taxa_ids, profile=args.profile, time_step=args.time_step,
        anchor_x=x, anchor_mu=anchor_mu, anchor_winner_idx=anchor_winner_idx,
        s_grid=s_grid,
        sweep_X=sweep_X, sweep_mu=sweep_mu,
        sweep_winner_idx=sweep_winner_idx, sweep_gap=sweep_gap,
        anchor_winner_gr_curve=anchor_winner_gr_curve,
        score_winner_gr_var=score_winner_gr_var, score_gap_var=score_gap_var,
        flip_taxon_j=np.array(flip_taxon_j), flip_grid_idx=np.array(flip_grid_idx),
        flip_from_idx=np.array(flip_from_idx), flip_to_idx=np.array(flip_to_idx),
        flip_s_j=np.array(flip_s_j),
        skipped_taxa=np.array(skipped_taxa),
        epsilon=args.epsilon, grid_points_m=M, seed=args.seed,
        checkpoint=str(args.checkpoint), lp_trajectories=str(args.lp_trajectories),
    )
    print(f"  Saved: {out_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    section("7. Plots")

    non_skipped = [j for j in ranked if taxa_ids[j] not in skipped_taxa]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    labels = [taxa_ids[j] for j in non_skipped]
    ax = axes[0]
    bar_colors = ["#D4A017" if j == anchor_winner_idx else "#5B8DB8" for j in non_skipped]
    bars = ax.bar(labels, [score_winner_gr_var[j] for j in non_skipped], color=bar_colors)
    for bar, j in zip(bars, non_skipped):
        n_flip = flip_counts[taxa_ids[j]]
        label_bits = []
        if j == anchor_winner_idx:
            label_bits.append("own share")
        if n_flip > 0:
            label_bits.append(f"{n_flip} flip{'s' if n_flip != 1 else ''}")
        if label_bits:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    " / ".join(label_bits),
                    ha="center", va="bottom", fontsize=7, color="firebrick", fontweight="bold")
    ax.set_ylabel("Var(anchor winner's growth rate)\nover taxon's own sweep")
    ax.set_title(f"Per-taxon sensitivity ranking -- anchor: {args.profile} @ t={args.time_step} "
                 f"(winner: {taxa_ids[anchor_winner_idx]}, gold bar = winner's own share effect)")

    ax = axes[1]
    ax.bar(labels, [score_gap_var[j] for j in non_skipped], color="#E07B54")
    ax.set_ylabel("Var(top1-top2 gap)\nover taxon's own sweep")
    ax.set_xlabel("Taxon (ranked by primary score)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)

    ranking_path = args.plots_dir / f"sensitivity_ranking_{args.profile}_t{args.time_step}.png"
    fig.savefig(ranking_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {ranking_path}")

    top_j = non_skipped[:args.top_n_curves]
    ncols = 3
    nrows = int(np.ceil(len(top_j) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             constrained_layout=True, squeeze=False)
    for idx, j in enumerate(top_j):
        ax = axes[idx // ncols][idx % ncols]
        sorted_mu = np.sort(sweep_mu[j], axis=1)
        ax.plot(s_grid, sorted_mu[:, -1], color="black", linewidth=1.6, label="current leader")
        ax.plot(s_grid, sorted_mu[:, -2], color="grey", linewidth=1.2, linestyle="--",
                label="runner-up")
        ax.plot(s_grid, anchor_winner_gr_curve[j], color="#5B8DB8", linewidth=1.8,
                label=f"{taxa_ids[anchor_winner_idx]} (anchor's winner)")
        for m, fs in zip(
            [g for t, g in zip(flip_taxon_j, flip_grid_idx) if t == taxa_ids[j]],
            [s for t, s in zip(flip_taxon_j, flip_s_j) if t == taxa_ids[j]],
        ):
            ax.axvline(fs, color="firebrick", linestyle=":", linewidth=1.0)
        ax.set_title(f"vary taxon {taxa_ids[j]}'s share", fontsize=10)
        ax.set_xlabel("share")
        ax.set_ylabel("growth rate")
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(top_j), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    curves_path = args.plots_dir / f"sensitivity_curves_{args.profile}_t{args.time_step}.png"
    fig.suptitle(f"Response curves -- top {len(top_j)} most sensitive taxa "
                 f"(anchor: {args.profile} @ t={args.time_step})", fontsize=12)
    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {curves_path}")

    # ── 8. Stratified oracle-validation candidate set ───────────────────────────
    section("8. Stratified oracle-validation candidate set")
    n_non_skipped = len(non_skipped)
    # ceil (not round) so the pooled candidate count meets or exceeds the
    # target before the random subsample-down step -- rounding down here would
    # silently under-deliver n_validation_points whenever it's close to n_taxa.
    k_per_taxon = max(1, -(-args.n_validation_points // n_non_skipped))
    candidate_rows = []
    for j in non_skipped:
        idxs = np.unique(np.linspace(0, M - 1, k_per_taxon).astype(int))
        for m in idxs:
            candidate_rows.append(sweep_X[j, m])
    candidate_rows = np.array(candidate_rows)
    candidate_rows = np.unique(candidate_rows, axis=0)

    rng = np.random.default_rng(args.seed)
    if len(candidate_rows) > args.n_validation_points:
        sel = rng.choice(len(candidate_rows), size=args.n_validation_points, replace=False)
        candidate_rows = candidate_rows[sel]
    print(f"  Selected {len(candidate_rows)} validation candidates "
          f"(target {args.n_validation_points}, stratified across all {n_non_skipped} taxa's sweeps)")

    cand_path = args.output_dir / f"validation_candidates_{args.profile}_t{args.time_step}.npz"
    np.savez_compressed(
        cand_path,
        X=candidate_rows, taxa_ids=taxa_ids, n=len(candidate_rows),
        seed=args.seed, sampling_mode="usecase_a_sensitivity_grid",
    )
    print(f"  Saved: {cand_path}")
    print(f"\n  Next: submit LP oracle validation, e.g.:")
    print(f"    sbatch --time=00:30:00 --array=0-0 \\")
    print(f"        --export=CANDIDATES_FILE={cand_path},N_SHARDS=1,"
          f"SHARD_SUBDIR=usecase_a_validation,BASE_SEED={args.seed} \\")
    print(f"        scripts/hvsc1_al_eval_candidates_array.sh")


if __name__ == "__main__":
    main()
