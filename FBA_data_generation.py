import numpy as np
import os

def generate_fba_data(model, vman, file_path=None):
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

    # Create a range of vman values
    vman_values = np.linspace(rxn.lower_bound, rxn.upper_bound, 1000)
    print(f"upper bound: {rxn.lower_bound}. lower bound: {rxn.upper_bound}")

    feasibility_dict = {}  # Tracks whether each vman value is feasible
    X, Y = [], []          # Feasible data points for training

    # Sweep through all vman values
    for v in vman_values:
        try:
            # Fix the flux value for the controlled reaction
            rxn.bounds = (v, v)
        except Exception:
            # If COBRA refuses the bound (e.g. inconsistent), mark as infeasible
            feasibility_dict[v] = False
            continue

        # Solve the FBA problem
        solution = model.optimize()

        # Record feasibility
        is_feasible = solution.status == "optimal"
        feasibility_dict[v] = is_feasible

        # Save data only if feasible
        if is_feasible:
            x = [v]
            y = [
                solution.fluxes.get("EX_etoh_e", 0.0),          # Ethanol exchange
                solution.fluxes.get("EX_glc__D_e", 0.0),        # Glucose exchange
                solution.fluxes.get("EX_co2_e", 0.0),           # CO2 exchange
                solution.fluxes.get("Biomass_Ecoli_core", 0.0), # Biomass
                # solution.fluxes.get("EX_ac_e"), # Acetate exchange
                # solution.fluxes.get("EX_h_e"), # H+ exchange
                # solution.fluxes.get("EX_h2o_e"), # H2O exchange
                # solution.fluxes.get("EX_lac__D_e"), # D-lactate exchange
                # solution.fluxes.get("EX_succ_e"), # Succinate exchange
            ]
            X.append(x)
            Y.append(y)
    
    if file_path != None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez_compressed(file_path, 
                    X=np.array(X), 
                    Y=np.array(Y), 
                    feasibility_dict=feasibility_dict)

    return X, Y, feasibility_dict

def list_infeasible_regions(feasibility_dict, rxn_id="PFK"):
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

