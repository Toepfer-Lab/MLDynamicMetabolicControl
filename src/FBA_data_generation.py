import numpy as np
import os

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
        List of corresponding output flux vectors [v_etoh, v_glc, v_co2, v_biomass].
    feasibility_dict : dict
        Dictionary mapping every sampled vman_value → True (feasible) or False (infeasible).
    """
    rxn = model.reactions.get_by_id(vman)

    # STEP 1
    # Do a coarse sweep of vman in order to identify feasible regions
    coarse_values = np.linspace(rxn.lower_bound, rxn.upper_bound, 200)
    feasibility_dict = {}  # Tracks whether each vman value is feasible
    feasible_points = []

    for v in coarse_values:
        rxn.bounds = (v, v)
        solution = model.optimize()
        feasible = solution.status == "optimal"
        feasibility_dict[v] = feasible
        if feasible:
            feasible_points.append(v)

    if not feasible_points:
        raise RuntimeError(f"No feasible steady states found for {vman}")
        
    # next we identify the feasible region
    # assuming that there is only one continous region with upper and lower limit
    feasible_min, feasible_max = min(feasible_points), max(feasible_points)
    print(f"Feasible region for {vman}: [{feasible_min:.2f}, {feasible_max:.2f}]")

    # STEP 2
    # now we sample with n_samples within our feasible region
    # this way we keep a consistent training point number across potentially different vman fluxes

    # Create a range of vman values
    vman_values = np.linspace(feasible_min, feasible_max, n_samples)
    feasible_range = (feasible_min, feasible_max)
    X, Y = [], []    #save data points in these lists

    for v in vman_values:
        rxn.bounds = (v, v)
        solution = model.optimize()
        if solution.status == "optimal":
            X.append([v])
            Y.append([
                solution.fluxes.get("EX_etoh_e", 0.0),
                solution.fluxes.get("EX_glc__D_e", 0.0),
                solution.fluxes.get("EX_co2_e", 0.0),
                solution.fluxes.get("Biomass_Ecoli_core", 0.0),
            ])
    
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez_compressed(file_path, 
                    X=np.array(X), 
                    Y=np.array(Y), 
                    feasible_range = feasible_range
                    )
    return X, Y, feasible_range

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

