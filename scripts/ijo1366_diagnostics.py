#!/usr/bin/env python
"""
Stage 1 diagnostics for the iJO1366 surrogate pipeline.

Confirms:
  1. INPUT_FLUX_IDS and OUTPUT_FLUX_IDS from flux_config_ijo1366 exist in the model
  2. FVA ranges for input fluxes (wide range = usable control knob)
  3. Each output flux actually varies across the input box (not pinned near zero)
  4. The two input fluxes are not tightly collinear

Review the printed summary before proceeding to Stage 2 (data generation).
Update OUTPUT_FLUX_IDS in src/flux_config_ijo1366.py if any outputs should be dropped.

Usage:
    python scripts/ijo1366_diagnostics.py
    python scripts/ijo1366_diagnostics.py --condition anaerobic
    python scripts/ijo1366_diagnostics.py --model-path model/iJO1366.xml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cobra.io import load_model, read_sbml_model, write_sbml_model
from cobra.flux_analysis import flux_variability_analysis

from flux_config_ijo1366 import INPUT_FLUX_IDS, OUTPUT_FLUX_IDS, SAMPLING_BOUNDS_OVERRIDE

N_VARIABILITY_SAMPLES = 50
COLLINEARITY_THRESHOLD = 0.9
CV_PINNED_THRESHOLD = 0.01  # std/|mean| below this → flag as likely pinned


def configure_medium(model, condition: str) -> None:
    """Glucose minimal medium, aerobic or anaerobic O2."""
    medium = dict(model.medium)

    # Set glucose uptake to 10 mmol/g/h (handle both naming conventions)
    exchange_ids = {r.id for r in model.exchanges}
    for glc_id in ("EX_glc__D_e", "EX_glc_D_e"):
        if glc_id in exchange_ids:
            medium[glc_id] = 10.0
            break

    if condition == "anaerobic":
        if "EX_o2_e" in medium:
            medium["EX_o2_e"] = 0.0

    model.medium = medium

    # Prefer the core biomass reaction as objective
    biomass_rxns = [r for r in model.reactions if "BIOMASS" in r.id]
    core_biomass = [r for r in biomass_rxns if "core" in r.id.lower()]
    model.objective = (core_biomass or biomass_rxns)[0].id


def check_reaction_ids(model, ids: list) -> tuple:
    found = [rid for rid in ids if rid in model.reactions]
    missing = [rid for rid in ids if rid not in model.reactions]
    return found, missing


def run_fva_for_inputs(model, input_ids: list):
    """FVA at fraction_of_optimum=0.0 to get the widest feasible range."""
    return flux_variability_analysis(
        model, reaction_list=input_ids, fraction_of_optimum=0.0
    )


def sample_variability(model, input_ids: list, output_ids: list, fva, n_samples: int, seed: int = 42):
    """
    Uniformly sample the FVA-bounded input box, solve LP for each point,
    collect output flux values and actual input flux values at each feasible point.
    Returns (out_vals dict, in_vals dict, n_feasible).
    """
    rng = np.random.default_rng(seed)

    lo = np.array([fva.loc[rid, "minimum"] for rid in input_ids])
    hi = np.array([fva.loc[rid, "maximum"] for rid in input_ids])

    # Guard against degenerate FVA ranges (should not happen, but be safe)
    narrow = (hi - lo) < 1e-6
    if narrow.any():
        ids_narrow = [input_ids[i] for i in np.where(narrow)[0]]
        print(f"  WARNING: near-zero FVA range for {ids_narrow}; expanding by ±0.5 for sampling")
        lo[narrow] -= 0.5
        hi[narrow] += 0.5

    samples = rng.uniform(lo, hi, size=(n_samples, len(input_ids)))

    out_vals = {oid: [] for oid in output_ids}
    in_vals = {iid: [] for iid in input_ids}
    n_feasible = 0

    for sample in samples:
        with model:
            for i, rid in enumerate(input_ids):
                model.reactions.get_by_id(rid).bounds = (float(sample[i]), float(sample[i]))
            sol = model.optimize()
            if sol.status == "optimal":
                n_feasible += 1
                for oid in output_ids:
                    out_vals[oid].append(sol.fluxes.get(oid, 0.0))
                for iid in input_ids:
                    in_vals[iid].append(sol.fluxes[iid])

    return out_vals, in_vals, n_feasible


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--condition", choices=["aerobic", "anaerobic"], default="aerobic",
        help="Medium condition for FVA and variability sampling (default: aerobic)",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to local SBML file; downloads iJO1366 from BiGG if not provided",
    )
    parser.add_argument(
        "--n-samples", type=int, default=N_VARIABILITY_SAMPLES,
        help=f"Random samples for output variability check (default: {N_VARIABILITY_SAMPLES})",
    )
    args = parser.parse_args()

    print("=" * 66)
    print("  iJO1366 surrogate pipeline  --  Stage 1 diagnostics")
    print(f"  Condition : {args.condition}")
    print(f"  Inputs    : {INPUT_FLUX_IDS}")
    print(f"  Outputs   : {OUTPUT_FLUX_IDS}")
    print("=" * 66)

    # ── Load model ────────────────────────────────────────────────────────
    local_path = REPO_ROOT / "model" / "iJO1366.xml"
    if args.model_path and args.model_path.exists():
        print(f"\nLoading model from {args.model_path} ...")
        model = read_sbml_model(str(args.model_path))
    elif local_path.exists():
        print(f"\nLoading cached model from {local_path} ...")
        model = read_sbml_model(str(local_path))
    else:
        print("\nDownloading iJO1366 from BiGG (cobra.io.load_model) ...")
        t0 = time.time()
        model = load_model("iJO1366")
        elapsed = time.time() - t0
        print(f"  Loaded in {elapsed:.1f}s  "
              f"({len(model.reactions)} reactions, {len(model.metabolites)} metabolites)")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        write_sbml_model(model, str(local_path))
        print(f"  Saved local copy -> {local_path}")

    configure_medium(model, args.condition)
    baseline = model.optimize()
    print(f"\nBaseline objective  : {model.objective}")
    print(f"Baseline value      : {baseline.objective_value:.6f} h^-1")

    # ── 1. Reaction ID existence ──────────────────────────────────────────
    print("\n── 1. Reaction ID existence check " + "─" * 32)
    found, missing = check_reaction_ids(model, INPUT_FLUX_IDS + OUTPUT_FLUX_IDS)
    for rid in INPUT_FLUX_IDS:
        status = "OK" if rid in found else "MISSING"
        print(f"  [INPUT ] {rid:<46}  {status}")
    for rid in OUTPUT_FLUX_IDS:
        status = "OK" if rid in found else "MISSING"
        print(f"  [OUTPUT] {rid:<46}  {status}")

    # Report all BIOMASS reactions for reference (important: confirm correct ID)
    biomass_ids = sorted(r.id for r in model.reactions if "BIOMASS" in r.id)
    print(f"\n  All BIOMASS reactions in model: {biomass_ids}")

    if missing:
        print(f"\n  WARNING: {len(missing)} ID(s) not found in model: {missing}")
        for rid in missing:
            near = [r.id for r in model.reactions if rid.lower() in r.id.lower()][:5]
            if near:
                print(f"    Possible matches for '{rid}': {near}")

    valid_inputs = [rid for rid in INPUT_FLUX_IDS if rid in model.reactions]
    valid_outputs = [rid for rid in OUTPUT_FLUX_IDS if rid in model.reactions]

    if not valid_inputs:
        print("\nERROR: No valid input flux IDs found. Cannot proceed.")
        sys.exit(1)

    # ── 2. FVA on input fluxes ────────────────────────────────────────────
    print(f"\n── 2. FVA ranges for input fluxes (condition: {args.condition}) " + "─" * 12)
    print("  (fraction_of_optimum=0.0  ->  widest feasible range)")
    t0 = time.time()
    fva = run_fva_for_inputs(model, valid_inputs)
    print(f"  Completed in {time.time() - t0:.1f}s\n")
    print(f"  {'Reaction':<22}  {'min [mmol/g/h]':>16}  {'max [mmol/g/h]':>16}  {'range':>10}")
    print("  " + "─" * 68)
    for rid in valid_inputs:
        lo, hi = fva.loc[rid, "minimum"], fva.loc[rid, "maximum"]
        flag = "  <-- narrow!" if (hi - lo) < 0.1 else ""
        print(f"  {rid:<22}  {lo:>16.4f}  {hi:>16.4f}  {hi - lo:>10.4f}{flag}")

    # Effective sampling bounds (FVA + SAMPLING_BOUNDS_OVERRIDE)
    print(f"\n  Effective sampling bounds for Stage 2 (after SAMPLING_BOUNDS_OVERRIDE):")
    print(f"  {'Reaction':<22}  {'eff min':>10}  {'eff max':>10}  note")
    print(f"  {'─'*52}")
    for rid in valid_inputs:
        fva_lo = fva.loc[rid, "minimum"]
        fva_hi = fva.loc[rid, "maximum"]
        ov = SAMPLING_BOUNDS_OVERRIDE.get(rid, (None, None))
        eff_lo = ov[0] if ov[0] is not None else fva_lo
        eff_hi = ov[1] if ov[1] is not None else fva_hi
        note = "clamped" if ov[0] is not None or ov[1] is not None else ""
        print(f"  {rid:<22}  {eff_lo:>10.4f}  {eff_hi:>10.4f}  {note}")

    # ── 3. Output flux variability ────────────────────────────────────────
    print(f"\n── 3. Output flux variability  ({args.n_samples} uniform samples in FVA box) " + "─" * 4)
    t0 = time.time()
    out_vals, in_vals, n_feasible = sample_variability(
        model, valid_inputs, valid_outputs, fva, args.n_samples
    )
    print(f"  Completed in {time.time() - t0:.1f}s  "
          f"|  feasible: {n_feasible}/{args.n_samples} "
          f"({100 * n_feasible / args.n_samples:.0f}%)\n")

    print(f"  {'Flux ID':<46}  {'min':>9}  {'max':>9}  {'std':>9}  {'CV':>8}  verdict")
    print("  " + "─" * 100)
    pinned_outputs = []
    for oid in valid_outputs:
        vals = np.array(out_vals[oid])
        if len(vals) == 0:
            print(f"  {oid:<46}  {'—':>9}  {'—':>9}  {'—':>9}  {'—':>8}  NO FEASIBLE SAMPLES")
            continue
        mn, mx, sd, mean = vals.min(), vals.max(), vals.std(), vals.mean()
        # Pinned-at-zero case: both std and range are negligible
        if sd < 1e-9 and (mx - mn) < 1e-9:
            cv_str = "0/0"
            pinned = True
        else:
            cv = sd / abs(mean) if abs(mean) > 1e-9 else float("inf")
            cv_str = f"{cv:.4f}"
            pinned = cv < CV_PINNED_THRESHOLD
        if pinned:
            pinned_outputs.append(oid)
        verdict = "LIKELY PINNED -- consider dropping" if pinned else "varies"
        print(f"  {oid:<46}  {mn:>9.4f}  {mx:>9.4f}  {sd:>9.4f}  {cv_str:>8}  {verdict}")

    # ── 4. Pairwise coupling check ────────────────────────────────────────
    if len(valid_inputs) >= 2:
        print(f"\n── 4. Pairwise coupling  ({valid_inputs[0]} vs {valid_inputs[1]}) " + "─" * 25)
        v1 = np.array(in_vals[valid_inputs[0]])
        v2 = np.array(in_vals[valid_inputs[1]])
        if len(v1) > 2 and v1.std() > 1e-9 and v2.std() > 1e-9:
            r = float(np.corrcoef(v1, v2)[0, 1])
            verdict = (
                "COLLINEAR -- consider dropping one"
                if abs(r) > COLLINEARITY_THRESHOLD
                else "independent"
            )
            print(f"  Pearson r = {r:+.4f}  ->  {verdict}")
        else:
            print("  Insufficient variation in sampled input fluxes to compute correlation.")

    # ── Summary ───────────────────────────────────────────────────────────
    fva_ok = all(
        (fva.loc[rid, "maximum"] - fva.loc[rid, "minimum"]) > 0.1
        for rid in valid_inputs
    )
    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    print(f"  Missing IDs          : {missing or 'none'}")
    print(f"  Input FVA ranges     : {'OK (non-trivial)' if fva_ok else 'WARNING -- narrow or zero range'}")
    print(f"  Feasibility rate     : {n_feasible}/{args.n_samples} = {100 * n_feasible / args.n_samples:.0f}%")
    print(f"  Likely-pinned outputs: {pinned_outputs or 'none'}")
    print()
    if pinned_outputs or missing:
        print("  ACTION: Update OUTPUT_FLUX_IDS in src/flux_config_ijo1366.py")
        print("          to remove pinned/missing IDs, then rerun to confirm.")
    else:
        print("  All checks passed. Proceed to Stage 2:")
        print("    python scripts/ijo1366_generate_data.py --n-samples 5000")


if __name__ == "__main__":
    main()
