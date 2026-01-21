#!/usr/bin/env python
"""
CLI to generate steady-state FBA training data for a chosen manipulated flux.
"""

import argparse
import sys
from typing import Optional
from pathlib import Path

from src.plotting import plot_flux_space

# Ensure local src is importable when running via sbatch
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from cobra.io import load_model, read_sbml_model


import FBA_data_generation  # noqa: E402
from runtime_utils import DATA_DIR, ensure_output_dirs  # noqa: E402


def configure_medium(model, condition: str, glucose_ub: Optional[float]):
    """
    Apply aerobic/anaerobic settings, optionally clamp glucose uptake,
    and relax selected exchange reactions.
    """
    model.reactions.get_by_id("EX_glyc_e").bounds = (0, 1000)
    model.reactions.get_by_id("EX_succ_e").bounds = (0, 1000)
    medium = model.medium


    solution = model.optimize()
    model.summary()
    
    print(f"ACKr value: {solution.fluxes['ACKr']}")
    print(f"glycine uptake: {solution.fluxes['EX_glyc_e']}")
    medium["EX_glc__D_e"] = 10.0  # default glucose uptake

    model.medium = medium
    print(medium)
    return medium


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vman", default="PYK", help="Reaction ID to manipulate (e.g. PFK, PYK)")


    parser.add_argument(
        "--model-id",
        default="textbook",
        help="COBRA model ID to load (e.g. 'textbook'); ignored if --model-path is set",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to SBML model file (overrides --model-id)",
    )
    parser.add_argument(
        "--condition",
        choices=["aerobic", "anaerobic"],
        default="anaerobic",
        help="Media condition to apply",
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples across feasible region")
    parser.add_argument(
        "--glucose-ub",
        type=float,
        default=None,
        help="Upper bound for glucose uptake (EX_glc__D_e); leave unset to keep model default",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save generated data (.npz). Defaults to data/fba_data_<vman>_<condition>.npz",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    print(f"model path is: {args.model_path}")
    output_path = args.output or DATA_DIR / f"fba_data_{args.vman}_{args.condition}.npz"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    mp = args.model_path
    mp = None if mp is None else str(mp).strip()

    if mp and mp.lower() not in {"none", "null"}:
        model = read_sbml_model(mp)
    else:
        model = load_model(args.model_id)

    configure_medium(model, args.condition, args.glucose_ub)

    X, Y, feasible_range = FBA_data_generation.generate_fba_data(
        model,
        args.vman,
        file_path=str(output_path),
        n_samples=args.n_samples,
    )

    print(f"Saved {len(X)} feasible points to {output_path}")
    print(f"Feasible region for {args.vman}: [{feasible_range[0]:.2f}, {feasible_range[1]:.2f}]")
    plot_flux_space(X, Y, feasible_range, vman_id=args.vman)

if __name__ == "__main__":
    main()
