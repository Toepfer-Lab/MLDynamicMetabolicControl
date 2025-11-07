import numpy as np
print(np.__version__)
import torch

def h(z):
    glucose = z[0]
    ethanol = z[1]
    ks = 2.964e-4   # substrate affinity (glucose)
    ki = 25         # inhibition constant (ethanol)
    return (glucose / (glucose + ks)) * (1 / (1 + ethanol / ki))

def hybrid_ode(t, z, vman_func, model_nn, x_scaler, y_scaler):
    """
    ODE system for hybrid model.
    
    Parameters
    ----------
    t : float
        Current time
    z : list of float
        Current concentrations [glucose, ethanol, biomass]
    vman_func : callable
        Function that returns control flux (vman) at time t
    model_nn : torch.nn.Module
        Trained neural network for predicting exchange fluxes
    x_scaler, y_scaler : StandardScaler
        Scalers used during training (for normalization)
    
    Returns
    -------
    dzdt : list of float
        Derivatives [d(glucose)/dt, d(ethanol)/dt, d(biomass)/dt]
    """

    #print(f"we're in the hybrid ODE and received vman_func: {vman_func}, that is of type: {type(vman_func)}")

    # Ensure non-negative concentrations
    z = np.maximum(z, 0)
    glucose, ethanol, biomass = z

    # Get control input value
    vman_val = vman_func(t)  # scalar
    vman_scaled = x_scaler.transform([[vman_val]])
    vman_tensor = torch.tensor(vman_scaled, dtype=torch.float32)

    # NN prediction of external fluxes
    with torch.no_grad():
        vext_scaled = model_nn(vman_tensor).numpy()
        vext = y_scaler.inverse_transform(vext_scaled).flatten()

    # Unpack predicted fluxes
    v_etoh, v_glc, v_co2, v_bio = vext

    # Dynamic rate scaling (e.g., Michaelis-Menten / biomass-based)
    rate = biomass * h(z)
    #print(f"rate: {rate}, h(z): {h(z)}, biomass: {biomass}")

    # Derivatives (glucose is consumed → negative)
    dzdt = np.array([
        rate * v_glc,   # d(glucose)/dt
        rate * v_etoh,   # d(ethanol)/dt
        rate * v_bio     # d(biomass)/dt
    ])

    dzdt[z <= 0] = np.maximum(dzdt[z <= 0], 0)

    return dzdt


