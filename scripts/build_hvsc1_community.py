"""
Build a MICOM community model for a synthetic community (default: HvSC1).

Mirrors the workflow in data/MICOM.ipynb: read a taxonomy table, build a
micom-format dataframe (id, file, abundance, ...), construct a Community,
pickle it, then sanity-check it with cooperative_tradeoff at a few fractions.

Strains can be pruned via --strains, so the community can be shrunk later
without touching this script.

Usage:
    python scripts/build_hvsc1_community.py
    python scripts/build_hvsc1_community.py --strains 100,161,163,352
"""

import argparse
import time
from pathlib import Path

import cobra
import pandas as pd
from micom import Community, load_pickle

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taxonomy-path", type=Path,
                    default=REPO_ROOT / "data" / "SynComs_taxonomy.csv")
    p.add_argument("--model-dir", type=Path,
                    default=REPO_ROOT / "data" / "HvSC1")
    p.add_argument("--syncom", type=str, default="HvSC1",
                    help="Value of the 'Syncom' column to filter taxonomy on")
    p.add_argument("--extra-syncoms", type=str, default="Both",
                    help="Additional Syncom values to also include (comma-separated). "
                         "Use '' to disable. Default: 'Both'")
    p.add_argument("--strains", type=str, default="all",
                    help="Comma-separated Strain IDs to include, or 'all' "
                         "for every strain with a matching model file")
    p.add_argument("--output", type=Path,
                    default=REPO_ROOT / "model" / "hvsc1_comm.pickle")
    p.add_argument("--solver", type=str, default="cplex")
    p.add_argument("--fractions", type=str, default="0,0.5,1",
                    help="Comma-separated cooperative_tradeoff fractions "
                         "for the diagnostic sanity check")
    return p.parse_args()


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


def apply_medium_and_unlock(comm):
    """Unlock internal exchanges (standard micom cross-feeding practice)."""
    for rxn in comm.internal_exchanges:
        rxn.bounds = (-1000.0, 1000.0)


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load taxonomy and match to available model files ─────────────────
    section("1. Loading taxonomy and matching to available models")
    taxonomy = pd.read_csv(args.taxonomy_path, dtype={"Strain ID": str})
    extra = [s.strip() for s in args.extra_syncoms.split(",") if s.strip()]
    all_syncoms = [args.syncom] + extra
    syncom_df = taxonomy[taxonomy["Syncom"].isin(all_syncoms)].copy()
    print(f"  Taxonomy rows with Syncom in {all_syncoms}: {len(syncom_df)}")

    # Strain ID is the filename prefix before the first underscore, e.g.
    # "100_or_mb1_mdr_rdr_dp_mb2_lib_bz_fix.xml" -> "100". Colleague's
    # corrected model delivery (2026-07-14) uses this naming scheme,
    # replacing the original "{id}_ex.xml" convention.
    model_files = {f.stem.split("_")[0]: f for f in args.model_dir.glob("*.xml")}
    print(f"  Model files found in {args.model_dir}: {len(model_files)}")

    csv_ids = set(syncom_df["Strain ID"])
    file_ids = set(model_files)

    missing_files = sorted(csv_ids - file_ids, key=int)
    missing_rows = sorted(file_ids - csv_ids, key=int)
    if missing_files:
        rows = syncom_df[syncom_df["Strain ID"].isin(missing_files)]
        print(f"  WARNING: {len(missing_files)} taxonomy row(s) have no "
              f"model file and will be dropped:")
        for _, row in rows.iterrows():
            print(f"    {row['Strain ID']:<6} {row['Species']}")
    if missing_rows:
        print(f"  WARNING: {len(missing_rows)} model file(s) have no "
              f"taxonomy row and will be dropped: {missing_rows}")

    available_ids = sorted(csv_ids & file_ids, key=int)
    print(f"  Available strains (taxonomy ∩ model files): {len(available_ids)}")

    # ── 2. Strain selection ───────────────────────────────────────────────
    section("2. Selecting strains")
    if args.strains == "all":
        selected_ids = available_ids
    else:
        requested_ids = [s.strip() for s in args.strains.split(",") if s.strip()]
        unknown_ids = sorted(set(requested_ids) - set(available_ids), key=int)
        if unknown_ids:
            print(f"  WARNING: requested strain(s) not available, "
                  f"skipping: {unknown_ids}")
        selected_ids = [s for s in requested_ids if s in available_ids]
    print(f"  Strains selected for community: {len(selected_ids)}")
    print(f"    {selected_ids}")

    # ── 3. Build the micom taxonomy dataframe ────────────────────────────
    section("3. Building the micom taxonomy dataframe")
    comm_df = syncom_df[syncom_df["Strain ID"].isin(selected_ids)].copy()
    comm_df = comm_df.rename(columns={
        "Strain ID": "id", "Genus": "genus", "Species": "species", "Family": "family",
    })[["id", "genus", "species", "family"]]
    comm_df["file"] = comm_df["id"].apply(lambda x: str(model_files[x]))
    comm_df["abundance"] = 1
    comm_df = comm_df.reset_index(drop=True)
    print(comm_df.to_string(index=False))

    # ── 4. Build and save the Community ──────────────────────────────────
    section(f"4. Building the Community ({len(comm_df)} taxa, solver={args.solver})")
    cobra.Configuration.solver = args.solver
    t0 = time.time()
    community = Community(
        comm_df,
        name=f"{args.syncom}_community",
        id=f"{args.syncom.lower()}_comm",
        solver=args.solver,
    )
    print(f"  Built in {time.time()-t0:.1f} s")

    community.to_pickle(str(args.output))
    print(f"  Saved: {args.output}")

    # ── 5. Diagnostic sanity check ────────────────────────────────────────
    section("5. Diagnostic cooperative_tradeoff sanity check")
    print("  No community-specific medium file exists yet for this syncom; "
          "the diagnostic below uses the merged models' own default "
          "exchange bounds.")
    comm = load_pickle(str(args.output))
    apply_medium_and_unlock(comm)

    fractions = [float(f) for f in args.fractions.split(",")]
    results = []
    for fraction in fractions:
        sol = comm.cooperative_tradeoff(fluxes=True, pfba=True, fraction=fraction)
        results.append((fraction, sol.status, float(sol.growth_rate)))
        print(f"  fraction={fraction:<4}  status={sol.status:<10}  "
              f"growth_rate={sol.growth_rate:.4f}")

    # ── 6. Summary ─────────────────────────────────────────────────────────
    section("6. Summary")
    print(f"  Strains included : {len(selected_ids)}")
    if missing_files:
        print(f"  Strains dropped (no model) : {missing_files}")
    print(f"  Output pickle    : {args.output}")
    print(f"  Diagnostic results:")
    for fraction, status, growth_rate in results:
        print(f"    fraction={fraction:<4}  status={status:<10}  growth_rate={growth_rate:.4f}")


if __name__ == "__main__":
    main()
