"""
Validate Use-case B's landscape sweep against real LP ground truth.

Joins the LP validation shard (written by hvsc1_convergence_sweep_array.sh /
hvsc1_convergence_sweep.py against the stratified subsample selected by
hvsc1_usecase_b_select_validation_ics.py) back to the original
million-point landscape sweep's surrogate-predicted values for the same
starting points, via the selected-indices mapping.

Reports:
  - Pearson/Spearman correlation between REAL (n_lp_calls) and
    surrogate-predicted convergence_time for the same points.
  - Final-dominant-taxon agreement rate between real LP and the surrogate's
    own (unperturbed) prediction.

Scope note: this does NOT ground-truth agreement_fraction itself (that would
need K real perturbed-LP trajectories per point, ~10x the cost here) -- see
--replicate-lp-shard for an optional smaller spot-check if one has been run.

Usage:
    python scripts/hvsc1_usecase_b_validate_compare.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lp-shard", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_convergence_sweep"
                           / "usecase_b_validation" / "shard00.npz")
    p.add_argument("--selected-indices", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b_validation_selected_indices.npz")
    p.add_argument("--landscape-npz", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b" / "landscape_sweep.npz")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b" / "validation_comparison.npz")
    p.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "plots" / "hvsc1_usecase_b")
    return p.parse_args()


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    section("1. Loading LP validation shard and surrogate-side landscape")
    shard = np.load(args.lp_shard, allow_pickle=True)
    sel = np.load(args.selected_indices, allow_pickle=True)
    landscape = np.load(args.landscape_npz, allow_pickle=True)
    taxa_ids = [str(t) for t in shard["taxa_ids"]]

    n = len(shard["n_lp_calls"])
    print(f"  LP shard: {n} validated ICs")

    real_convergence_time = shard["n_lp_calls"]
    real_final_dominant_idx = np.argmax(shard["lp_trajectories"][:, -1, :], axis=1)
    # This shard's own fresh (unperturbed) surrogate re-run of the same ICs --
    # the direct, same-methodology comparator for real_final_dominant_idx.
    surrogate_shard_final_dominant_idx = np.argmax(shard["surrogate_trajectories"][:, -1, :], axis=1)

    # Join back to the big landscape sweep via the global_indices mapping
    # (row i of this 40-point validation batch -> global_indices[i] in the
    # 1,000,000-row landscape_sweep.npz).
    global_indices = sel["global_indices"]
    landscape_convergence_time = landscape["convergence_time"][global_indices]
    landscape_final_dominant_idx = landscape["final_dominant_idx"][global_indices]
    landscape_agreement_fraction = landscape["agreement_fraction"][global_indices]

    section("2. Convergence-time correlation (real LP vs. surrogate landscape sweep)")
    rho_p, p_p = pearsonr(real_convergence_time, landscape_convergence_time)
    rho_s, p_s = spearmanr(real_convergence_time, landscape_convergence_time)
    print(f"  Pearson  r = {rho_p:.4f}  (p={p_p:.2e})")
    print(f"  Spearman rho = {rho_s:.4f}  (p={p_s:.2e})")
    mae_conv = float(np.mean(np.abs(real_convergence_time - landscape_convergence_time)))
    print(f"  MAE(convergence_time) = {mae_conv:.2f} steps")

    section("3. Final-dominant-taxon agreement (real LP vs. surrogate)")
    agree_landscape = float(np.mean(real_final_dominant_idx == landscape_final_dominant_idx))
    agree_shard = float(np.mean(real_final_dominant_idx == surrogate_shard_final_dominant_idx))
    print(f"  Real LP vs. landscape-sweep surrogate prediction : {agree_landscape:.4f} ({n} points)")
    print(f"  Real LP vs. this shard's own fresh surrogate run : {agree_shard:.4f} ({n} points) "
          f"(sanity check -- should be very close to the row above)")

    section("4. Scope note")
    print("  agreement_fraction (the K-replicate instability signal) is NOT ground-truthed here --")
    print("  that would require K real PERTURBED LP trajectories per point (~10x this run's cost).")
    print("  Only convergence_time and the single unperturbed final-dominant prediction are validated.")

    section("5. Saving")
    np.savez_compressed(
        args.output,
        global_indices=global_indices,
        real_convergence_time=real_convergence_time,
        landscape_convergence_time=landscape_convergence_time,
        real_final_dominant_idx=real_final_dominant_idx,
        landscape_final_dominant_idx=landscape_final_dominant_idx,
        landscape_agreement_fraction=landscape_agreement_fraction,
        pearson_r=rho_p, pearson_p=p_p, spearman_rho=rho_s, spearman_p=p_s,
        mae_convergence_time=mae_conv,
        agreement_rate_vs_landscape=agree_landscape,
        agreement_rate_vs_shard_own_surrogate=agree_shard,
        taxa_ids=taxa_ids,
    )
    print(f"  Saved: {args.output}")

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(real_convergence_time, landscape_convergence_time, s=40, alpha=0.7, color="#5B8DB8")
    lims = [0, max(real_convergence_time.max(), landscape_convergence_time.max()) + 2]
    ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Real LP convergence time (steps)")
    ax.set_ylabel("Surrogate-predicted convergence time (steps)")
    ax.set_title(f"Use-case B validation (n={n})\n"
                 f"Pearson r={rho_p:.3f}, Spearman rho={rho_s:.3f}, MAE={mae_conv:.2f} steps\n"
                 f"final-dominant agreement={agree_landscape:.2%}")
    path = args.plots_dir / "validation_convergence_time_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
