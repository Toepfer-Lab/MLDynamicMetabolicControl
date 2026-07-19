"""
Visualise the Use-case B landscape sweep (results/hvsc1_usecase_b/landscape_sweep.npz).

Produces (in --plots-dir, default plots/hvsc1_usecase_b/):
  convergence_vs_agreement_scatter.png   -- the direct hypothesis-test figure:
      convergence_time vs. agreement_fraction, one point per starting
      composition. Tests whether "stable states function as switches that
      regulate community growth speed" -- i.e. whether points near a decision
      boundary (low agreement_fraction) also take longer to converge.
  landscape_projection_convergence_time.png
  landscape_projection_agreement_fraction.png
      -- 2D spatial scatter of starting compositions in (taxon_x, taxon_y)
      space, colored continuously by each metric (style matching the existing
      full_basin_map.png convention, but a continuous colormap instead of
      discrete per-taxon colors since these are continuous metrics).

Usage:
    python scripts/hvsc1_plot_landscape.py
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
    p.add_argument("--landscape-npz", type=Path,
                   default=REPO_ROOT / "results" / "hvsc1_usecase_b" / "landscape_sweep.npz")
    p.add_argument("--taxon-x", type=str, default="946")
    p.add_argument("--taxon-y", type=str, default="644")
    p.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "plots" / "hvsc1_usecase_b")
    return p.parse_args()


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    section("1. Loading landscape sweep")
    d = np.load(args.landscape_npz, allow_pickle=True)
    taxa_ids = [str(t) for t in d["taxa_ids"]]
    x0 = d["x0"]
    convergence_time = d["convergence_time"]
    agreement_fraction = d["agreement_fraction"]
    n = len(convergence_time)
    print(f"  N = {n}   sweep_wall_time_seconds = {float(d['sweep_wall_time_seconds']):.2f}")
    print(f"  convergence_time: mean={convergence_time.mean():.2f} median={np.median(convergence_time):.0f} "
          f"max={convergence_time.max()}")
    print(f"  agreement_fraction: mean={agreement_fraction.mean():.4f} min={agreement_fraction.min():.4f}")

    section("2. Testing the switch hypothesis (convergence_time vs. agreement_fraction)")
    rho_p, p_p = pearsonr(convergence_time, agreement_fraction)
    rho_s, p_s = spearmanr(convergence_time, agreement_fraction)
    print(f"  Pearson  r = {rho_p:.4f}  (p={p_p:.2e})")
    print(f"  Spearman rho = {rho_s:.4f}  (p={p_s:.2e})")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(convergence_time, agreement_fraction, s=3, alpha=0.15, color="#5B8DB8")
    ax.set_xlabel("Convergence time (steps to reach 0.99 abundance)")
    ax.set_ylabel("Agreement fraction (K jittered replicates matching\nthe unperturbed final dominant taxon)")
    ax.set_title(f"Convergence time vs. local stability (n={n})\n"
                 f"Pearson r={rho_p:.3f}, Spearman rho={rho_s:.3f}")
    path = args.plots_dir / "convergence_vs_agreement_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    section("3. Spatial projections")
    if args.taxon_x not in taxa_ids or args.taxon_y not in taxa_ids:
        raise SystemExit(f"--taxon-x/--taxon-y must be in {taxa_ids}")
    ix, iy = taxa_ids.index(args.taxon_x), taxa_ids.index(args.taxon_y)

    for metric_name, metric, cmap in [
        ("convergence_time", convergence_time, "viridis"),
        ("agreement_fraction", agreement_fraction, "RdYlGn"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        sc = ax.scatter(x0[:, ix], x0[:, iy], c=metric, s=3, alpha=0.4, cmap=cmap)
        fig.colorbar(sc, ax=ax, label=metric_name)
        ax.set_xlabel(f"taxon {args.taxon_x} abundance")
        ax.set_ylabel(f"taxon {args.taxon_y} abundance")
        ax.set_title(f"Landscape projection colored by {metric_name} (n={n})")
        path = args.plots_dir / f"landscape_projection_{metric_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
