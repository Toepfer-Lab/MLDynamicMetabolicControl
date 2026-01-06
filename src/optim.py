import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def objective_fn(sol):
    return sol.y[2, -1]  # biomass final


def piecewise_constant_control(control_times, values):
    values = np.asarray(values)

    def control(t):
        idx = np.searchsorted(control_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return control


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
    verbose=False,
    log_full_every_k=10,   # store full trajectories every k evals
    log_best_full=True,    # also store full trajectory whenever we hit a new best
    
):
    control_times = np.linspace(t_span[0], t_span[1], N + 1)    #create time grid
    logs = []

    eval_idx = 0
    best_final = -np.inf

    def simulate(vman_values):
        nonlocal eval_idx, best_final

        vman_t = piecewise_constant_control(control_times, vman_values)

        def rhs(t, z):
            return hybrid_ode(t, z, vman_t, model, x_scaler, y_scaler)

        sol = solve_ivp(rhs, t_span, z0, t_eval=t_eval_points, method="RK45")

        # Always log light info (if solver succeeded)
        if sol.success:
            final_biomass = sol.y[2, -1]

            store_full = (log_full_every_k is not None and log_full_every_k > 0 and (eval_idx % log_full_every_k == 0))

            is_new_best = final_biomass > best_final
            if is_new_best:
                best_final = final_biomass
                if log_best_full:
                    store_full = True  # override: keep the curve at new best

            entry = {
                "eval_idx": eval_idx,
                "vman_values": np.copy(vman_values),
                "final_biomass": float(final_biomass),
                "store_full": bool(store_full),
            }

            if store_full:
                entry["t"] = sol.t
                entry["biomass"] = sol.y[2, :]

            logs.append(entry)

        eval_idx += 1
        return sol

    if initial_guess is None:
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        initial_guess = (lower_bounds + upper_bounds) / 2.0

    def cost(vman_values):
        sol = simulate(vman_values)
        if sol is None or not sol.success:
            return np.inf
        final_biomass = objective_fn(sol)
        if verbose:
            print(f"[eval {eval_idx-1}] Candidate biomass: {final_biomass:.4f}")
        return -final_biomass

    result = minimize(
        cost,
        x0=np.asarray(initial_guess, dtype=float),
        bounds=bounds,
        method="powell",
        options={"maxiter": 100000, "ftol": 1e-4},
    )

    return result, logs
