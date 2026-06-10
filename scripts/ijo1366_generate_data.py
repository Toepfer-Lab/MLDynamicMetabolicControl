#!/usr/bin/env python
"""
Stage 2: Generate N-D FBA training data for the iJO1366 surrogate.

Samples the 2-D (ACKr, LDH_D) input box uniformly under anaerobic conditions,
solves the LP for each point, and records feasible output fluxes.
Sampling bounds come from FVA with SAMPLING_BOUNDS_OVERRIDE applied on top.

Infeasibility rate and per-output statistics are printed as diagnostics.
To be reviewed before proceeding to Stage 3 (surrogate training).

Usage:
    python scripts/ijo1366_generate_data.py
    python scripts/ijo1366_generate_data.py --n-samples 5000
    python scripts/ijo1366_generate_data.py --n-samples 100   # quick smoke test
"""


import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cobra.flux_analysis import flux_variability_analysis
from cobra.io import load_model, read_sbml_model

from FBA_data_generation import generate_fba_data_nd
from flux_config_ijo1366 import (
    INPUT_FLUX_IDS,
    OUTPUT_FLUX_IDS,
    OUTPUT_LABELS,
    SAMPLING_BOUNDS_OVERRIDE,
)
from runtime_utils import DATA_DIR, ensure_output_dirs


def configure_medium(model, condition: str) -> None:
    """Glucose minimal medium, aerobic or anaerobic O2. Mirrors diagnostics setup."""
    medium = dict(model.medium)
    exchange_ids = {r.id for r in model.exchanges}
    for glc_id in ("EX_glc__D_e", "EX_glc_D_e"):
        if glc_id in exchange_ids:
            medium[glc_id] = 10.0
            break
    if condition == "anaerobic" and "EX_o2_e" in medium:
        medium["EX_o2_e"] = 0.0
    model.medium = medium
    biomass_rxns = [r for r in model.reactions if "BIOMASS" in r.id]
    core_biomass = [r for r in biomass_rxns if "core" in r.id.lower()]
    model.objective = (core_biomass or biomass_rxns)[0].id


def compute_effective_bounds(model, input_ids, overrides):
    """
    FVA at fraction_of_optimum=0.0, then apply SAMPLING_BOUNDS_OVERRIDE.
    Prints a table of FVA vs effective bounds and returns list of (lo, hi).
    """
    print("  Running FVA for sampling bounds ...")
    fva = flux_variability_analysis(model, reaction_list=input_ids, fraction_of_optimum=0.0)

    print(f"\n  {'Reaction':<22}  {'FVA min':>10}  {'FVA max':>10}  "
          f"{'eff min':>10}  {'eff max':>10}  note")
    print(f"  {'─'*72}")
    bounds = []
    for rid in input_ids:
        fva_lo = fva.loc[rid, "minimum"]
        fva_hi = fva.loc[rid, "maximum"]
        ov = overrides.get(rid, (None, None))
        eff_lo = ov[0] if ov[0] is not None else fva_lo
        eff_hi = ov[1] if ov[1] is not None else fva_hi
        note = "clamped" if ov[0] is not None or ov[1] is not None else ""
        print(f"  {rid:<22}  {fva_lo:>10.4f}  {fva_hi:>10.4f}  "
              f"{eff_lo:>10.4f}  {eff_hi:>10.4f}  {note}")
        bounds.append((eff_lo, eff_hi))
    return bounds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Total samples drawn (default: 5000)")
    parser.add_argument("--condition", choices=["aerobic", "anaerobic"], default="anaerobic",
                        help="Medium condition (default: anaerobic)")
    parser.add_argument("--model-path", type=Path,
                        default=REPO_ROOT / "model" / "iJO1366.xml",
                        help="Path to SBML model (default: model/iJO1366.xml)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .npz path (default: data/ijo1366_<condition>.npz)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    output_path = args.output or DATA_DIR / f"ijo1366_{args.condition}.npz"

    print("=" * 66)
    print("  iJO1366 surrogate pipeline  --  Stage 2: data generation")
    print(f"  Condition  : {args.condition}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  Inputs     : {INPUT_FLUX_IDS}")
    print(f"  Outputs    : {OUTPUT_FLUX_IDS}")
    print(f"  Output     : {output_path}")
    print("=" * 66)

    # Load model
    if args.model_path.exists():
        print(f"\nLoading model from {args.model_path} ...")
        model = read_sbml_model(str(args.model_path))
    else:
        print("\nLocal model not found; downloading iJO1366 from BiGG ...")
        model = load_model("iJO1366")

    configure_medium(model, args.condition)
    baseline = model.optimize()
    print(f"Baseline biomass: {baseline.objective_value:.4f} h^-1\n")

    # Compute effective sampling bounds (FVA + clamp)
    print("── Sampling bounds ─────────────────────────────────────────────")
    bounds = compute_effective_bounds(model, INPUT_FLUX_IDS, SAMPLING_BOUNDS_OVERRIDE)

    # Generate data
    print(f"\n── Generating {args.n_samples} samples ─────────────────────────────────────")
    t0 = time.time()
    X, Y, X_infeasible, infeasibility_rate = generate_fba_data_nd(
        model,
        vman_ids=INPUT_FLUX_IDS,
        output_flux_ids=OUTPUT_FLUX_IDS,
        output_labels=OUTPUT_LABELS,
        bounds=bounds,
        n_samples=args.n_samples,
        file_path=str(output_path),
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(f"\n── Summary ─────────────────────────────────────────────────────")
    print(f"  Wall time      : {elapsed:.1f}s  ({1000 * elapsed / args.n_samples:.1f} ms/sample)")
    print(f"  X (feasible)   : {X.shape}")
    print(f"  Y              : {Y.shape}")
    print(f"  X_infeasible   : {X_infeasible.shape}  (saved for boundary analysis)")
    print(f"  Saved to       : {output_path}")
    print()
    print(f"  Infeasibility  : {100 * infeasibility_rate:.1f}%  "
          f"({'OK — expected joint-constraint corner effects' if infeasibility_rate <= 0.25 else 'HIGH — consider tightening SAMPLING_BOUNDS_OVERRIDE'})")
    print()
    print("  Proceed to Stage 3:")
    print(f"    python scripts/train_surrogate.py --data-path {output_path} "
          f"--hidden-dim 16 --vman ijo1366")


if __name__ == "__main__":
    main()
