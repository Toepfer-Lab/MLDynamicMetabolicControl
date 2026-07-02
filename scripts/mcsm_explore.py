"""
Community model structure explorer.

Runs on the cluster with CPLEX loaded. Answers all structural questions
needed before designing a surrogate:
  - Community taxonomy (who is in it, how many organisms)
  - Medium composition (which metabolites, how many, current bounds)
  - Exchange reaction structure (community vs. internal)
  - A single cooperative_tradeoff solve to inspect output shape/content
  - Timing of one solve

Output is plain text to stdout so it lands in the SLURM log.
"""

import sys
import time
import pandas as pd
import cobra
import micom
from micom import load_pickle

cobra.Configuration.solver = "cplex"

MODEL_PATH  = "/home/jkaatz/MA/MLDynamicMetabolicControl/model/dcom.pickle"
MEDIUM_PATH = "/home/jkaatz/MA/MLDynamicMetabolicControl/data/Completed_maize_leaf_medium.csv"
FRACTION    = 0.5   # safe midpoint for the test solve


def section(title):
    print(f"\n{'='*66}")
    print(f"  {title}")
    print(f"{'='*66}")


# ── 1. Load ───────────────────────────────────────────────────────────────────
section("1. Loading community model")
t0 = time.time()
comm = load_pickle(MODEL_PATH)
print(f"  Loaded in {time.time()-t0:.1f} s")
print(f"  Model ID : {comm.id}")


# ── 2. Taxonomy ───────────────────────────────────────────────────────────────
section("2. Community taxonomy")
taxa = comm.taxa
print(f"  Number of taxa : {len(taxa)}")
print(f"  Taxonomy columns: {list(comm.taxonomy.columns) if hasattr(comm, 'taxonomy') else 'N/A'}")
print(f"\n  First 10 taxa:")
for t in list(taxa)[:10]:
    print(f"    {t}")
if len(taxa) > 10:
    print(f"    ... ({len(taxa)-10} more)")


# ── 3. Scale ──────────────────────────────────────────────────────────────────
section("3. Model scale")
rxns  = len(comm.reactions)
mets  = len(comm.metabolites)
print(f"  Total reactions   : {rxns:,}")
print(f"  Total metabolites : {mets:,}")


# ── 4. Medium ─────────────────────────────────────────────────────────────────
section("4. Medium composition — applying colleague's maize leaf medium")
medium_df = pd.read_csv(MEDIUM_PATH, index_col=0)
print(f"  CSV rows: {len(medium_df)}  columns: {list(medium_df.columns)}")

# Build dict and apply — MICOM ignores IDs not present in model
medium_dict = dict(zip(medium_df["reaction"], medium_df["flux"]))
comm.medium = medium_dict
medium = comm.medium   # read back what was actually accepted

print(f"  Requested : {len(medium_dict)} components")
print(f"  Accepted  : {len(medium)} (only IDs present in model are kept)")
print(f"\n  Active medium (non-zero bounds):")
for rxn_id, ub in sorted(medium.items()):
    if ub > 0:
        print(f"    {rxn_id:<45}  {ub:.6f}")

# Report any requested IDs that weren't in the model
missing = [k for k in medium_dict if k not in comm.medium and medium_dict[k] > 0]
if missing:
    print(f"\n  IDs in CSV not found in model ({len(missing)}):")
    for m in missing:
        print(f"    {m}")


# ── 5. Exchange reactions ─────────────────────────────────────────────────────
section("5. Exchange reaction structure")

# Community-level exchanges (connect community to environment)
comm_exchanges = [r for r in comm.exchanges]
print(f"  Community exchange reactions : {len(comm_exchanges)}")

# Internal exchanges (connect individual organisms to shared medium)
int_exchanges = list(comm.internal_exchanges)
print(f"  Internal exchange reactions  : {len(int_exchanges)}")
print(f"\n  First 10 internal exchange IDs:")
for r in int_exchanges[:10]:
    print(f"    {r.id:<55}  bounds={r.bounds}")
if len(int_exchanges) > 10:
    print(f"    ... ({len(int_exchanges)-10} more)")

# Current bounds on internal exchanges
lb_vals = [r.lower_bound for r in int_exchanges]
ub_vals = [r.upper_bound for r in int_exchanges]
print(f"\n  Internal exchange bound summary:")
print(f"    lower bounds — min={min(lb_vals):.1f}, max={max(lb_vals):.1f}")
print(f"    upper bounds — min={min(ub_vals):.1f}, max={max(ub_vals):.1f}")


# ── 6. Abundance ──────────────────────────────────────────────────────────────
section("6. Abundance values")
try:
    abundances = comm.taxonomy["abundance"]
    print(f"  Unique abundance values: {sorted(abundances.unique())}")
    print(f"  Abundance stats: min={abundances.min():.3f}  max={abundances.max():.3f}  "
          f"mean={abundances.mean():.3f}")
except Exception as e:
    print(f"  Could not read abundances: {e}")


# ── 7. Single cooperative_tradeoff solve ──────────────────────────────────────
section(f"7. Single cooperative_tradeoff solve  (fraction={FRACTION})")

# Unlock internal exchanges (standard MICOM practice)
for rxn in comm.internal_exchanges:
    rxn.bounds = (-1000.0, 1000.0)
print(f"  Internal exchanges unlocked to (-1000, 1000)")

print(f"  Running cooperative_tradeoff(fraction={FRACTION}, pfba=True, fluxes=True) ...")
t0 = time.time()
sol = comm.cooperative_tradeoff(fraction=FRACTION, pfba=True, fluxes=True)
elapsed = time.time() - t0
print(f"  Solve time : {elapsed:.1f} s")
print(f"  Status     : {sol.status if hasattr(sol, 'status') else 'N/A'}")


# ── 8. CommunitySolution introspection ────────────────────────────────────────
section("8. CommunitySolution — available attributes")
all_attrs = [a for a in dir(sol) if not a.startswith("_")]
print(f"  All non-private attributes/methods:")
for a in all_attrs:
    val = getattr(sol, a, None)
    kind = type(val).__name__
    try:
        shape = val.shape if hasattr(val, "shape") else (len(val) if hasattr(val, "__len__") else "—")
    except Exception:
        shape = "?"
    print(f"    {a:<30}  type={kind:<20}  size/shape={shape}")


# ── 9. Output structure ───────────────────────────────────────────────────────
section("9. Output structure")

# Community-level scalar growth rate
print(f"  growth_rate (community scalar) : {sol.growth_rate:.8f}")
print(f"  objective_value                : {sol.objective_value:.8f}")

# Per-taxon rates are in sol.members
members = sol.members
print(f"\n  sol.members  : shape={members.shape}  columns={list(members.columns)}")
print(f"\n  Full members DataFrame:")
print(members.to_string())

# Extract per-taxon growth rate column (usually 'growth_rate')
gr_col = [c for c in members.columns if "growth" in c.lower()]
if gr_col:
    gr = members[gr_col[0]]
    print(f"\n  Per-taxon growth rates (column '{gr_col[0]}'):")
    for taxon_id, rate in gr.items():
        print(f"      {taxon_id:<20}  {rate:.8f}")
    print(f"  Non-zero taxa: {(gr > 1e-9).sum()} / {len(gr)}")
else:
    print("  No growth rate column found in members.")

# Full flux DataFrame (only present when fluxes=True)
fl = sol.fluxes
print(f"\n  fluxes DataFrame : shape={fl.shape}  (rows=taxa, cols=reactions)")
print(f"    Index (taxa)   : {list(fl.index)}")
print(f"    Columns sample : {list(fl.columns[:5])} ... {list(fl.columns[-3:])}")

# Community-level exchange fluxes — reactions ending in __m or EX_*_m
ex_cols = [c for c in fl.columns if c.endswith("__m") or c.endswith("_m")]
print(f"\n  Community exchange columns in fluxes : {len(ex_cols)}")

if ex_cols:
    ex_sub = fl[ex_cols]
    # Sum across taxa to get net community flux per metabolite
    net = ex_sub.sum(axis=0)
    secreted = net[net > 1e-9].sort_values(ascending=False)
    consumed  = net[net < -1e-9].sort_values()
    print(f"\n  Net secreted by community (top 15, positive = leaving community):")
    for rxn, val in secreted.head(15).items():
        print(f"    {rxn:<50}  {val:+.6f}")
    print(f"\n  Net consumed by community (top 15, negative = taken up):")
    for rxn, val in consumed.head(15).items():
        print(f"    {rxn:<50}  {val:+.6f}")

# Check for exchange_fluxes attribute directly on solution
if hasattr(sol, "exchange_fluxes"):
    ef = sol.exchange_fluxes
    print(f"\n  sol.exchange_fluxes : shape={ef.shape}")


# ── 10. Summary for surrogate design ─────────────────────────────────────────
section("10. Surrogate design summary")
print(f"  Community size    : {len(comm.taxa)} taxa")
print(f"  LP scale          : {len(comm.reactions):,} reactions, {len(comm.metabolites):,} metabolites")
print(f"  Solve time        : {elapsed:.1f} s per cooperative_tradeoff call")
print(f"  Medium components : {len(medium)} (all bounds = 1000 → fully permissive)")
print(f"")
print(f"  Input axes identified:")
print(f"    fraction          : scalar [0, 1]         (1 dim — clean sweep)")
print(f"    medium bounds     : {len(medium)} components but all=1000  (need to decide which to vary)")
print(f"    abundance         : {len(comm.taxa)} values, NOT uniform        (could be an axis or fixed)")
print(f"")
print(f"  Output targets:")
print(f"    growth_rate       : {len(comm.taxa)} values (one per taxon)      ← compact, tractable")
print(f"    community exchange: {len(ex_cols)} values (net secretion/uptake) ← biologically focal")
print(f"    full fluxes       : {fl.shape[0]} × {fl.shape[1]} = {fl.shape[0]*fl.shape[1]:,}  ← too large")
print(f"")
print(f"  NOTE: All medium bounds are 1000 (unlimited). This means medium is not")
print(f"  currently constraining anything. The meaningful input axes are fraction")
print(f"  and/or specific medium bounds set to realistic values.")


# ── 11. set_abundance sensitivity test ───────────────────────────────────────
import inspect
import numpy as np

section("11. set_abundance — API inspection and sensitivity test")

# ── 11a. Does the method exist and what does it expect? ───────────────────────
if not hasattr(comm, "set_abundance"):
    print("  ERROR: comm.set_abundance does not exist on this MICOM version.")
    print("  Available methods:", [m for m in dir(comm) if "abund" in m.lower()])
else:
    print("  comm.set_abundance exists.")
    try:
        sig = inspect.signature(comm.set_abundance)
        print(f"  Signature: set_abundance{sig}")
        doc = comm.set_abundance.__doc__
        if doc:
            print(f"  Docstring (first 5 lines):")
            for line in doc.strip().splitlines()[:5]:
                print(f"    {line}")
    except Exception as e:
        print(f"  Could not inspect signature: {e}")

    # ── 11b. Build test profiles ──────────────────────────────────────────────
    taxa_ids = list(comm.taxa)
    n        = len(taxa_ids)
    rng      = np.random.default_rng(42)

    # Original abundances from the pickle taxonomy
    orig_vals = comm.taxonomy.set_index("id")["abundance"].reindex(taxa_ids).values

    profiles = {
        "original" : orig_vals,
        "uniform"  : np.ones(n) / n,
        "dirichlet_1": rng.dirichlet(np.ones(n)),
        "dirichlet_2": rng.dirichlet(np.ones(n)),
        "dirichlet_3": rng.dirichlet(np.ones(n)),
        "dominant_0" : np.array([0.95] + [0.05/(n-1)]*(n-1)),  # taxon 0 dominates
    }

    print(f"\n  Test profiles (n_taxa={n}, all rows sum to 1.0):")
    header = f"  {'profile':<16}" + "".join(f"  {t:<10}" for t in taxa_ids)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, vals in profiles.items():
        row = f"  {name:<16}" + "".join(f"  {v:<10.4f}" for v in vals)
        print(row)

    # ── 11c. Solve at each profile, collect growth rates ─────────────────────
    print(f"\n  Running cooperative_tradeoff(fraction={FRACTION}) for each profile ...")
    results = {}

    for name, vals in profiles.items():
        abundance_series = pd.Series(dict(zip(taxa_ids, vals)))
        t0 = time.time()
        try:
            comm.set_abundance(abundance_series)
            # Re-apply medium and unlock exchanges after set_abundance
            # (some MICOM versions rebuild internal state on abundance change)
            comm.medium = medium_dict
            for rxn in comm.internal_exchanges:
                rxn.bounds = (-1000.0, 1000.0)
            sol_i = comm.cooperative_tradeoff(fraction=FRACTION, pfba=False, fluxes=False)
            elapsed_i = time.time() - t0
            status = sol_i.status
            members_i = sol_i.members
            gr_col_i  = [c for c in members_i.columns if "growth" in c.lower()]
            gr_vals   = members_i[gr_col_i[0]].values if gr_col_i else np.full(n, np.nan)
            comm_rate = sol_i.growth_rate
        except Exception as e:
            elapsed_i = time.time() - t0
            status    = f"ERROR: {e}"
            gr_vals   = np.full(n, np.nan)
            comm_rate = np.nan

        results[name] = {
            "growth_rates" : gr_vals,
            "community_gr" : comm_rate,
            "time_s"       : elapsed_i,
            "status"       : status,
        }
        print(f"    {name:<16}  status={status:<10}  comm_gr={comm_rate:.6f}  "
              f"time={elapsed_i:.1f}s")

    # ── 11d. Comparison table ─────────────────────────────────────────────────
    print(f"\n  Per-taxon growth rates across profiles:")
    header = f"  {'profile':<16}" + "".join(f"  {t:<12}" for t in taxa_ids) + "  comm_gr"
    print(header)
    print("  " + "-" * len(header))
    for name, res in results.items():
        gr_str = "".join(f"  {v:<12.6f}" for v in res["growth_rates"])
        print(f"  {name:<16}{gr_str}  {res['community_gr']:.6f}")

    # ── 11e. Sensitivity summary ──────────────────────────────────────────────
    print(f"\n  Growth rate range across all profiles (max - min per taxon):")
    all_gr = np.array([r["growth_rates"] for r in results.values()])
    for i, t in enumerate(taxa_ids):
        lo, hi = np.nanmin(all_gr[:, i]), np.nanmax(all_gr[:, i])
        print(f"    {t:<20}  [{lo:.6f}, {hi:.6f}]  spread={hi-lo:.6f}")

    flat_range = np.nanmax(all_gr) - np.nanmin(all_gr)
    print(f"\n  Overall growth rate spread across all taxa and profiles: {flat_range:.6f}")
    if flat_range < 1e-6:
        print("  *** WARN: growth rates are invariant to abundance changes. "
              "Surrogate input via set_abundance may not be useful. ***")
    else:
        print("  Growth rates do vary with abundance → set_abundance is a viable input axis.")
