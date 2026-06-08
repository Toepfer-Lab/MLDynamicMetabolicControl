"""
Flux ID configuration for the iJO1366 genome-scale E. coli model.

INPUT_FLUX_IDS:  reactions pinned by the outer optimiser (the control knobs).
                 This is the single place to add a third control flux
                 (e.g. "PTAr", "PFL") without touching any other file.

OUTPUT_FLUX_IDS: reactions predicted by the surrogate (what the NN learns).
                 Run Stage 1 diagnostics to confirm all are non-trivially
                 variable; drop any that are flagged as "likely pinned".
"""

# Outer-optimiser-controlled fluxes — extend here only
INPUT_FLUX_IDS = ["ACKr", "LDH_D"]

# Surrogate output fluxes — update after Stage 1 diagnostic report if needed.
# EX_glc__D_e removed: glucose uptake is medium-constrained (always -10) and
# is handled by h(z) in the ODE, not predicted by the surrogate.
OUTPUT_FLUX_IDS = [
    "BIOMASS_Ec_iJO1366_core_53p95M",
    "EX_ac_e",
    "EX_co2_e",
    "EX_etoh_e",
]

OUTPUT_LABELS = ["bio", "ac", "co2", "etoh"]
OUTPUT_INDEX = {label: idx for idx, label in enumerate(OUTPUT_LABELS)}

# State variables for the hybrid ODE (confirmed in Stage 4)
STATE_LABELS = ("glucose", "biomass")
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_LABELS)}

# Sampling bound overrides applied on top of FVA ranges.
# Use None to keep the FVA value for that side.
# ACKr lower bound: FVA returns -1000 (model default box constraint, not biology).
# Clamped to -20 to match LDH_D scale and exclude unrealistic acetate uptake.
SAMPLING_BOUNDS_OVERRIDE = {
    "ACKr": (-20.0, None),
}
