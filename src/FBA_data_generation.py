import sys
import numpy as np
import os

if 'ipykernel' in sys.modules:
    # Running in Jupyter Notebook
    try:
        from flux_config import FLUX_LABELS, FLUX_RXN_IDS
    except ImportError:
        from src.flux_config import FLUX_LABELS, FLUX_RXN_IDS
else:
    # Running as a standard Python script
    try:
        from flux_config import FLUX_LABELS, FLUX_RXN_IDS
    except ImportError:
        from src.flux_config import FLUX_LABELS, FLUX_RXN_IDS

def generate_fba_data(model, vman, file_path=None, n_samples=1000):
    """
    Generate steady-state FBA data for a given control flux (vman).

    This function sweeps through a range of values for a specified reaction
    (the manipulated variable, `vman`) and records feasible steady-state fluxes
    of selected external metabolites and the biomass flux.

    Parameters
    ----------
    model : cobra.Model
        The COBRApy metabolic model.
    vman : str
        The reaction ID of the manipulated intracellular flux (e.g. "PFK").
    TODO: add number of vman values and which exchange fluxes to output

    Returns
    -------
    X : list of [float]
        List of feasible vman values (each wrapped in a list for ML compatibility).
    Y : list of [float]
        List of corresponding output flux vectors ordered by FLUX_RXN_IDS.
    feasibility_dict : dict
        Dictionary mapping every sampled vman_value → True (feasible) or False (infeasible).
    """
    rxn = model.reactions.get_by_id(vman)

    # first we check if all of the rxn_ids are actually avaialable in the model
    missing = [rid for rid in FLUX_RXN_IDS if rid not in model.reactions]
    if missing:
        raise KeyError(
            "These " + str(len(missing)) + " FLUX_RXN_IDS are not in the model: "
            + ", ".join(missing)
            + "\nAvailable exchanges: "
            + ", ".join([r.id for r in model.exchanges][:20])
        )

    # STEP 1
    # Do a coarse sweep of vman in order to identify feasible regions

    coarse_values = np.linspace(-100.0, 100.0, 2000)
    print(f"reaction bounds: [{rxn.lower_bound}, {rxn.upper_bound}]")
    feasibility_dict = {}  # Tracks whether each vman value is feasible
    feasible_points = []

    print(f"MODEL OBJECTIVE: {model.objective}")

    for v in coarse_values:
        rxn.bounds = (v, v)
        solution = model.optimize()
        feasible = solution.status == "optimal"
        feasibility_dict[v] = feasible
        if feasible:
            feasible_points.append(v)
    print("Feasibility sweep completed.")
    if not feasible_points:
        raise RuntimeError(f"No feasible steady states found for {vman}")
        
    # next we identify the feasible region
    # assuming that there is only one continous region with upper and lower limit
    feasible_min, feasible_max = min(feasible_points), max(feasible_points)
    print(f"Feasible region for {vman}: [{feasible_min:.2f}, {feasible_max:.2f}]")

    # STEP 2
    # now we sample with n_samples within our feasible region
    # this way we keep a consistent training point number across potentially different vman fluxes
    #feasible_min = 0.0
    #feasible_max = 10.0



    # Create a range of vman values
    vman_values = np.linspace(feasible_min, feasible_max, n_samples)
    feasible_range = (feasible_min, feasible_max)
    X, Y = [], []    #save data points in these lists

    #model.summary()

    for v in vman_values:
        rxn.bounds = (v, v)
        solution = model.optimize()
        if solution.status == "optimal":
            X.append([v])
            Y.append([solution.fluxes.get(rxn_id, 0.0) for rxn_id in FLUX_RXN_IDS])
    
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez_compressed(
            file_path,
            X=np.array(X),
            Y=np.array(Y),
            feasible_range=feasible_range,
            flux_order=np.array(FLUX_RXN_IDS),
            flux_labels=np.array(FLUX_LABELS),
        )
    return X, Y, feasible_range

def generate_fba_data_nd(model, vman_ids, output_flux_ids, bounds, n_samples,
                          output_labels=None, file_path=None, seed=42):
    """
    N-D naive uniform sampling for surrogate training data.

    Unlike generate_fba_data (1-D with FVA pre-masking), this function:
    - Accepts a list of control flux IDs and explicit sampling bounds
    - Samples the box naively; infeasibility rate is measured and reported
    - Uses `with model:` context for safe multi-flux pinning

    Parameters
    ----------
    model : cobra.Model
    vman_ids : list of str
        Reaction IDs to pin as surrogate inputs.
    output_flux_ids : list of str
        Reaction IDs to record as surrogate outputs.
    bounds : list of (float, float)
        Sampling box; one (lo, hi) tuple per entry in vman_ids.
    n_samples : int
        Total draws (includes infeasible; reported as diagnostic).
    output_labels : list of str or None
        Human-readable labels for output_flux_ids (for .npz metadata).
    file_path : str or None
        Path to save .npz; skipped if None.
    seed : int

    Returns
    -------
    X : np.ndarray, shape (n_feasible, n_inputs)
    Y : np.ndarray, shape (n_feasible, n_outputs)
    infeasibility_rate : float
    """
    missing_in = [r for r in vman_ids if r not in model.reactions]
    missing_out = [r for r in output_flux_ids if r not in model.reactions]
    if missing_in or missing_out:
        raise KeyError(
            f"Missing input IDs: {missing_in}  |  Missing output IDs: {missing_out}"
        )

    if output_labels is None:
        output_labels = output_flux_ids

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    samples = rng.uniform(lo, hi, size=(n_samples, len(vman_ids)))

    X, Y = [], []
    n_infeasible = 0
    report_every = max(1, n_samples // 10)

    for i, sample in enumerate(samples):
        with model:
            for j, rid in enumerate(vman_ids):
                model.reactions.get_by_id(rid).bounds = (float(sample[j]), float(sample[j]))
            sol = model.optimize()
            if sol.status == "optimal":
                X.append(sample.tolist())
                Y.append([sol.fluxes.get(oid, 0.0) for oid in output_flux_ids])
            else:
                n_infeasible += 1

        if (i + 1) % report_every == 0:
            pct = 100 * (i + 1) / n_samples
            print(f"  {i+1:>6}/{n_samples}  ({pct:.0f}%)  infeasible so far: {n_infeasible}")

    X = np.array(X) if X else np.empty((0, len(vman_ids)))
    Y = np.array(Y) if Y else np.empty((0, len(output_flux_ids)))
    infeasibility_rate = n_infeasible / n_samples

    print(f"\n  Total drawn    : {n_samples}")
    print(f"  Feasible       : {len(X)}")
    print(f"  Infeasibility  : {100 * infeasibility_rate:.1f}%")

    if len(Y) > 0:
        print(f"\n  Per-output statistics (unscaled):")
        print(f"  {'Flux ID':<46}  {'min':>9}  {'max':>9}  {'std':>9}  {'mean':>9}")
        print(f"  {'─'*90}")
        for k, oid in enumerate(output_flux_ids):
            col = Y[:, k]
            print(f"  {oid:<46}  {col.min():>9.4f}  {col.max():>9.4f}  "
                  f"{col.std():>9.4f}  {col.mean():>9.4f}")

    if file_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        np.savez_compressed(
            file_path,
            X=X,
            Y=Y,
            feasible_range=np.column_stack([lo, hi]),   # (n_inputs, 2) — generalised from 1-D
            flux_order=np.array(output_flux_ids),        # compatible with train_surrogate.py
            flux_labels=np.array(output_labels),
            input_flux_ids=np.array(vman_ids),
            n_samples_total=n_samples,
            infeasibility_rate=infeasibility_rate,
        )
        print(f"\n  Saved to {file_path}")

    return X, Y, infeasibility_rate


def list_infeasible_regions(feasibility_dict, rxn_id):
    """
    Identify continuous infeasible regions for a given manipulated reaction.

    Parameters
    ----------
    feasibility_dict : dict
        Dictionary mapping each vman value to True (feasible) or False (infeasible).
    rxn_id : str, optional
        The name or ID of the reaction being analyzed, used for printing.

    Returns
    -------
    infeasible_regions : list of (float, float)
        List of tuples defining start and end points of infeasible regions.
    """
    infeasible_regions = []
    sorted_v = sorted(feasibility_dict.keys())  # Ensure vman values are ordered
    start = None  # Marks the start of an infeasible block

    for i, v in enumerate(sorted_v):
        is_feasible = feasibility_dict[v]

        if not is_feasible:
            # Start of an infeasible region
            if start is None:
                start = v
        else:
            # End of an infeasible region
            if start is not None:
                infeasible_regions.append((start, sorted_v[i - 1]))
                start = None

    # Handle case where the last region extends to the end of the range
    if start is not None:
        infeasible_regions.append((start, sorted_v[-1]))

    # Display results
    if infeasible_regions:
        print(f"Infeasible regions for {rxn_id}:")
        for r in infeasible_regions:
            print(f"  [{r[0]:.2f}, {r[1]:.2f}]")
    else:
        print(f"No infeasible regions detected for {rxn_id}.")

    return infeasible_regions

def is_feasible(vman, infeasible_regions):
    for lower, upper in infeasible_regions:
        if lower <= vman <= upper:
            return False
    return True
