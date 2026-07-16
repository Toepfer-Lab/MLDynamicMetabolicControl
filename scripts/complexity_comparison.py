"""
Cross-scale comparison: surrogate speedup grows with model complexity.

Consolidates existing (and newly produced, see ecc2comp_benchmark_lp/surrogate.py)
LP-vs-surrogate benchmark results across four model scales — ECC2comp (~120
reactions), iJO1366 (2583 reactions), the mcsm 6-taxon MICOM community
(~16,400 reactions), and the hvsc1 27-taxon MICOM community (~70,800
reactions) — into one table and one log-scale figure. This is the
quantitative backbone for the thesis's core claim: the surrogate's absolute
benefit over a real LP solve grows with the complexity of the underlying
model, culminating in the largest MICOM community case.

hvsc1 was excluded for most of this project (see git history / calculations_log.md
2026-07-14 "hvsc1 removed from the comparison"): the originally-built model had
only 8,257 reactions -- smaller than the mcsm 6-taxon community despite having
more than 4x the taxa, which was itself part of the evidence that the model was
built incorrectly. A colleague delivered corrected source data, the community
was rebuilt (70,775 reactions / 46,476 metabolites, results/hvsc1_model_size.json),
the surrogate was retrained on a corner-bias-corrected dataset after a separate
diagnosis (calculations_log.md 2026-07-15/2026-07-16), and scripts/hvsc1_benchmark.py
(mirroring mcsm_benchmark.py) now provides the matched-call-count timing hvsc1
was missing. It is included here as the fourth and largest-scale point.

Methodology varies across the four scales included here (matched-call-count
benchmark throughout, but ECC2comp/iJO1366 are grid sweeps while mcsm/hvsc1 are
real 5-trajectory/80-step comparisons) — the methodology column in the output
table is not decorative, it is the whole point of being honest about what is
and isn't a like-for-like number.

Outputs:
  results/complexity_comparison.csv
  results/complexity_comparison.md
  plots/complexity_comparison/speedup_vs_complexity.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
PLOTS = REPO_ROOT / "plots" / "complexity_comparison"
PLOTS.mkdir(parents=True, exist_ok=True)

C_LP = "#E07B54"   # warm orange
C_SUR = "#5B8DB8"  # steel blue


def section(title):
    print(f"\n{'='*66}\n  {title}\n{'='*66}")


# ── 1. ECC2comp ────────────────────────────────────────────────────────────────
section("1. ECC2comp (matched grid_n=10, per-call aggregate)")
ecc_lp = np.load(RESULTS / "ecc2comp_benchmark_lp.npz")
ecc_sur = np.load(RESULTS / "ecc2comp_benchmark_surrogate.npz")

ecc_real_per_call = float(ecc_lp["sim_times"].sum() / ecc_lp["lp_calls"].sum())
ecc_sur_per_call = float(ecc_sur["sim_times"].sum() / ecc_sur["nn_calls"].sum())
ecc_speedup = ecc_real_per_call / ecc_sur_per_call
print(f"  Real (LP)  per call : {ecc_real_per_call*1000:.4f} ms")
print(f"  Surrogate  per call : {ecc_sur_per_call*1000:.4f} ms")
print(f"  Speedup             : {ecc_speedup:,.1f}x")

# ── 2. iJO1366 (piecewise variant only — the properly grid-matched one) ───────
section("2. iJO1366 (piecewise, matched grid_n=10, per-call aggregate)")
ijo_lp = np.load(RESULTS / "ijo1366_benchmark_piecewise_lp.npz")
ijo_sur = np.load(RESULTS / "ijo1366_benchmark_piecewise_surrogate.npz")

ijo_real_per_call = float(ijo_lp["sim_times"].sum() / ijo_lp["lp_calls"].sum())
ijo_sur_per_call = float(ijo_sur["sim_times"].sum() / ijo_sur["nn_calls"].sum())
ijo_speedup = ijo_real_per_call / ijo_sur_per_call
print(f"  Real (LP)  per call : {ijo_real_per_call*1000:.4f} ms")
print(f"  Surrogate  per call : {ijo_sur_per_call*1000:.4f} ms")
print(f"  Speedup             : {ijo_speedup:,.1f}x")

# ── 3. mcsm (6-taxon, real matched 5-trajectory/80-step comparison) ───────────
section("3. mcsm 6-taxon community (matched 5-trajectory/80-step, per-call)")
mcsm = np.load(RESULTS / "mcsm_benchmark.npz")

n_steps = int(mcsm["n_steps"])
mcsm_real_per_call = float(mcsm["lp_step_times"].mean())
mcsm_sur_per_call = float(mcsm["sur_traj_times_matched"].mean() / n_steps)
mcsm_speedup = mcsm_real_per_call / mcsm_sur_per_call
print(f"  Real (LP)  per call : {mcsm_real_per_call*1000:.4f} ms")
print(f"  Surrogate  per call : {mcsm_sur_per_call*1000:.4f} ms")
print(f"  Speedup             : {mcsm_speedup:,.1f}x")

# ── 4. hvsc1 (27-taxon, real matched 5-trajectory/80-step comparison) ─────────
section("4. hvsc1 27-taxon community (matched 5-trajectory/80-step, per-call)")
hvsc1 = np.load(RESULTS / "hvsc1_benchmark.npz")

hvsc1_n_steps = int(hvsc1["n_steps"])
hvsc1_real_per_call = float(hvsc1["lp_step_times"].mean())
hvsc1_sur_per_call = float(hvsc1["sur_traj_times_matched"].mean() / hvsc1_n_steps)
hvsc1_speedup = hvsc1_real_per_call / hvsc1_sur_per_call
print(f"  Real (LP)  per call : {hvsc1_real_per_call*1000:.4f} ms")
print(f"  Surrogate  per call : {hvsc1_sur_per_call*1000:.4f} ms")
print(f"  Speedup             : {hvsc1_speedup:,.1f}x")

# ── 5. mcsm basin-sensitivity capstone (reconciled, callout not a table row) ──
section("5. mcsm basin-sensitivity capstone (reconciled)")
capstone = np.load(RESULTS / "mcsm_basin_sensitivity_reconciled.npz")
print(f"  n_lp_solves      : {int(capstone['n_lp_solves']):,}")
print(f"  Surrogate time   : {float(capstone['surrogate_time_s'])/60:.1f} min")
print(f"  LP-estimate time : {float(capstone['lp_time_est_s'])/86400/365.25:.2f} years "
      f"(using measured {float(capstone['lp_step_time_s']):.3f} s/step)")
print(f"  Speedup          : {float(capstone['speedup']):,.0f}x")

# ── Assemble table ─────────────────────────────────────────────────────────────
section("6. Assembling comparison table")

rows = [
    dict(model="ECC2comp", n_rxn=122, n_met=94,
         real_ms=ecc_real_per_call * 1000, sur_ms=ecc_sur_per_call * 1000,
         speedup=ecc_speedup,
         methodology="Matched call-count (grid_n=10 sweep over ACKr, "
                     "real LP solve vs. NN forward pass at every ODE step)"),
    dict(model="iJO1366", n_rxn=2583, n_met=1805,
         real_ms=ijo_real_per_call * 1000, sur_ms=ijo_sur_per_call * 1000,
         speedup=ijo_speedup,
         methodology="Matched call-count (piecewise-control benchmark, "
                     "grid_n=10x10 over (ACKr, LDH_D))"),
    dict(model="mcsm (6-taxon)", n_rxn=16406, n_met=10592,
         real_ms=mcsm_real_per_call * 1000, sur_ms=mcsm_sur_per_call * 1000,
         speedup=mcsm_speedup,
         methodology="Matched call-count (5 real trajectories x 80 steps, "
                     "real cooperative_tradeoff vs. surrogate, both measured)"),
    dict(model="hvsc1 (27-taxon)", n_rxn=70775, n_met=46476,
         real_ms=hvsc1_real_per_call * 1000, sur_ms=hvsc1_sur_per_call * 1000,
         speedup=hvsc1_speedup,
         methodology="Matched call-count (5 real trajectories x 80 steps, "
                     "real cooperative_tradeoff vs. surrogate, both measured; "
                     "surrogate retrained on corner-bias-corrected data, see "
                     "calculations_log.md 2026-07-16)"),
]

csv_path = RESULTS / "complexity_comparison.csv"
with open(csv_path, "w") as f:
    f.write("model,n_reactions,n_metabolites,real_ms_per_call,surrogate_ms_per_call,speedup,methodology\n")
    for r in rows:
        f.write(f'"{r["model"]}",{r["n_rxn"]},{r["n_met"]},{r["real_ms"]:.4f},'
                f'{r["sur_ms"]:.5f},{r["speedup"]:.1f},"{r["methodology"]}"\n')
print(f"  Saved: {csv_path}")

md_path = RESULTS / "complexity_comparison.md"
with open(md_path, "w") as f:
    f.write("# Surrogate speedup vs. model complexity\n\n")
    f.write("| Model | #Reactions | #Metabolites | Real (ms/call) | Surrogate (ms/call) | Speedup | Methodology |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['model']} | {r['n_rxn']:,} | {r['n_met']:,} | {r['real_ms']:.4f} | "
                f"{r['sur_ms']:.5f} | {r['speedup']:,.1f}x | {r['methodology']} |\n")
    f.write(f"\n**Basin-sensitivity capstone (mcsm, reconciled)**: "
            f"{int(capstone['n_lp_solves']):,} LP-solve-equivalents, "
            f"surrogate {float(capstone['surrogate_time_s'])/60:.1f} min vs. "
            f"LP-estimate {float(capstone['lp_time_est_s'])/86400/365.25:.2f} years "
            f"-> {float(capstone['speedup']):,.0f}x speedup "
            f"(using the measured {float(capstone['lp_step_time_s']):.3f} s/step rate, "
            f"not the original 0.2 s/step assumption — see results/calculations_log.md).\n")
print(f"  Saved: {md_path}")

# ── Figure ──────────────────────────────────────────────────────────────────────
section("7. Plotting")

x = np.array([r["n_rxn"] for r in rows])
y_real = np.array([r["real_ms"] / 1000 for r in rows])   # back to seconds
y_sur = np.array([r["sur_ms"] / 1000 for r in rows])

fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

ax.plot(x, y_real, "o-", color=C_LP, linewidth=2, markersize=9, label="Real solve (LP / community LP)")
ax.plot(x, y_sur, "o-", color=C_SUR, linewidth=2, markersize=9, label="Surrogate (NN forward pass)")

for r, xi, yi_r in zip(rows, x, y_real):
    ax.annotate(f"{r['speedup']:,.0f}x", xy=(xi, yi_r), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Model complexity (#reactions, log scale)", fontsize=11)
ax.set_ylabel("Time per solve/call (s, log scale)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v*1000:.2f} ms" if v < 1 else f"{v:.1f} s"
))
ax.set_title("Surrogate speedup grows with model complexity\n"
             "(ECC2comp -> iJO1366 -> mcsm -> hvsc1 MICOM communities)",
             fontsize=11)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, which="both", alpha=0.2)

fig_path = PLOTS / "speedup_vs_complexity.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"  Saved: {fig_path}")

print("\nDone.")
