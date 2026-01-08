"""
Shared ordering for flux outputs and state vectors.
Keep this consistent across data generation, training, and simulation.
"""

# Order of flux outputs in Y (by reaction ID).
FLUX_RXN_IDS = (
    "EX_etoh_e",
    "EX_glc__D_e",
    "EX_co2_e",
    "Biomass_Ecoli_core",
)

# Human-friendly labels aligned with FLUX_RXN_IDS.
FLUX_LABELS = ("etoh", "glc", "co2", "biomass")
FLUX_INDEX = {name: idx for idx, name in enumerate(FLUX_LABELS)}

# Order of state variables in the hybrid ODE system.
STATE_LABELS = ("glucose", "ethanol", "biomass")
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_LABELS)}
