import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def objective_fn(sol):
    """
    Objective: maximize final biomass concentration.
    """
    biomass_final = sol.y[2, -1]  # assumes index 2 corresponds to biomass
    return biomass_final


def piecewise_constant_control(control_times, values):
    """
    Build a piecewise-constant control signal.
    """
    values = np.asarray(values)

    def control(t):
        idx = np.searchsorted(control_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return control


# Backwards compatibility for older imports
_piecewise_constant = piecewise_constant_control


def optimize_vman(
    model,
    hybrid_ode,
    z0,
    t_span,
    N,
    t_eval_points,
    bounds,
    x_scaler,
    y_scaler,
    initial_guess=None,
    log_trajectories=False,
    verbose=False,
):
    """
    Optimize the vman control trajectory to maximize biomass.
    Optionally logs every candidate trajectory and its biomass evolution.

    Parameters
    ----------
    model : torch.nn.Module
        Trained surrogate network.
    hybrid_ode : callable
        Right-hand-side function for the hybrid ODE.
    z0 : list
        Initial state [glucose, ethanol, biomass]
    t_span : tuple
        Time interval (t0, tf)
    N : int
        Number of control intervals
    t_eval_points : array
        Time points for solver output
    bounds : list of tuples
        Bounds for vman values
    x_scaler, y_scaler : sklearn scalers
        Normalization scalers used for the surrogate model.
    initial_guess : array-like, optional
        Starting guess for optimization; defaults to midpoint of bounds.
    log_trajectories : bool
        If True, stores each evaluated control trajectory and biomass evolution
    verbose : bool
        If True, print diagnostics during optimization.

    Returns
    -------
    result : OptimizeResult
        Output of scipy.optimize.minimize
    logs : list of dict (if log_trajectories=True)
        Each dict has keys:
            'vman_values' → np.array of control nodes
            't'           → time vector
            'biomass'     → biomass trajectory
            'final_biomass' → final biomass value
    """

    control_times = np.linspace(t_span[0], t_span[1], N + 1)  # split into N intervals
    logs = []

    def simulate(vman_values):
        vman_t = piecewise_constant_control(control_times, vman_values)

        def rhs(t, z):
            return hybrid_ode(t, z, vman_t, model, x_scaler, y_scaler)

        sol = solve_ivp(rhs, t_span, z0, t_eval=t_eval_points, method="RK45")

        if log_trajectories and sol.success:
            logs.append(
                {
                    "vman_values": np.copy(vman_values),
                    "t": sol.t,
                    "biomass": sol.y[2, :],
                    "final_biomass": sol.y[2, -1],
                }
            )

        return sol

    if initial_guess is None:
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        initial_guess = (lower_bounds + upper_bounds) / 2.0

    def cost(vman_values):
        sol = simulate(vman_values)
        if sol is None or not sol.success:
            return np.inf  # penalize integration failures
        final_biomass = objective_fn(sol)
        if verbose:
            print(f"Candidate biomass: {final_biomass:.4f}")
        return -final_biomass

    result = minimize(
        cost,
        x0=np.asarray(initial_guess, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
        options={"eps": 1e-1, "maxiter": 1000, "ftol": 1e-8},
    )

    if log_trajectories:
        return result, logs
    return result
