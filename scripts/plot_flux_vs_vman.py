#!/usr/bin/env python
"""
Sweep a vman range and plot FBA ethanol and biomass fluxes.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Non-interactive backend for cluster/headless runs
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Make local src importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cobra.io import load_model, read_sbml_model  # noqa: E402

from flux_config import FLUX_INDEX, FLUX_RXN_IDS  # noqa: E402
from runtime_utils import PLOT_DIR, ensure_output_dirs  # noqa: E402


def configure_medium(model, condition: str, glucose_ub: Optional[float]):
    """
    Apply aerobic/anaerobic settings and optionally clamp glucose uptake.
    """
    medium = model.medium
    if condition == "anaerobic":
        if "EX_o2_e" in medium:
            medium["EX_o2_e"] = 0.0
    if glucose_ub is not None:
        try:
            rxn = model.reactions.get_by_id("EX_glc__D_e")
            rxn.upper_bound = glucose_ub
        except KeyError:
            print("Warning: EX_glc__D_e not found; glucose bound unchanged.")
    model.medium = medium
    return medium


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vman",
        default="ACKr",
        help="Reaction ID to manipulate (e.g. PFK, PYK)",
    )
    p.add_argument(
        "--model-id",
        default="textbook",
        help="COBRA model ID to load (e.g. 'textbook'); ignored if --model-path is set",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to SBML model file (overrides --model-id)",
    )
    p.add_argument(
        "--condition",
        choices=["aerobic", "anaerobic"],
        default="anaerobic",
        help="Media condition to apply",
    )
    p.add_argument(
        "--glucose-ub",
        type=float,
        default=None,
        help="Upper bound for glucose uptake (EX_glc__D_e); leave unset to keep model default",
    )
    p.add_argument("--vmin", type=float, required=True, help="Lower bound for vman sweep")
    p.add_argument("--vmax", type=float, required=True, help="Upper bound for vman sweep")
    p.add_argument("--num-points", type=int, default=100, help="Number of vman values")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output plot path (default: plots/vman_flux_sweep_<vman>.png)",
    )
    p.add_argument("--show", action="store_true", help="Display plot interactively")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()

    mp = args.model_path
    mp = None if mp is None else str(mp).strip()
    if mp and mp.lower() not in {"none", "null"}:
        model = read_sbml_model(mp)
    else:
        model = load_model(args.model_id)

    configure_medium(model, args.condition, args.glucose_ub)

    try:
        rxn = model.reactions.get_by_id(args.vman)
    except KeyError as exc:
        raise SystemExit(f"Reaction {args.vman} not found in model.") from exc

    vman_id = args.vman

    vman_values = np.linspace(args.vmin, args.vmax, args.num_points)

    etoh_idx = FLUX_INDEX["etoh"]
    bio_idx = FLUX_INDEX["biomass"]
    etoh_rxn = FLUX_RXN_IDS[etoh_idx]
    bio_rxn = FLUX_RXN_IDS[bio_idx]

    etoh_flux = np.full_like(vman_values, np.nan, dtype=float)
    bio_flux = np.full_like(vman_values, np.nan, dtype=float)
    infeasible = 0

    for i, vman in enumerate(vman_values):
        rxn.bounds = (vman, vman)
        solution = model.optimize()
        if solution.status != "optimal":
            infeasible += 1
            continue
        etoh_flux[i] = float(solution.fluxes.get(etoh_rxn, 0.0))
        bio_flux[i] = float(solution.fluxes.get(bio_rxn, 0.0))

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(vman_values, etoh_flux, label="Ethanol flux", linewidth=2)
    plt.plot(vman_values, bio_flux, label="Biomass flux", linewidth=2)
    plt.xlabel(f"{vman_id} value")
    plt.ylabel("Flux")
    plt.title(f"FBA flux sweep vs {vman_id}")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = args.out or (PLOT_DIR / f"vman_flux_sweep_{vman_id}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    if infeasible:
        print(f"Skipped {infeasible} infeasible vman points.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
