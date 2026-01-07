import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution


def objective_fn(sol):
    return sol.y[2, -1]  # biomass final


def piecewise_constant_control(control_times, values):
    values = np.asarray(values)

    def control(t):
        idx = np.searchsorted(control_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return control


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
    log_full_every_k=10,
    log_best_full=True,
    global_maxiter=80,          # DE generations
    global_popsize=10,          # DE population size multiplier
    topk_polish=5,              # number of coarse points to be refined into a final result using powell
    polish_maxiter=2000,        # Powell iterations in polish phase
    seed=0,
):
    control_times = np.linspace(t_span[0], t_span[1], N + 1)
    logs = []

    eval_idx = 0
    best_final = -np.inf

    # Cache to avoid re-solving ODE for identical / near-identical points
    cache = {}
    def _cache_key(x, ndigits=10):
        return tuple(np.round(np.asarray(x, dtype=float), ndigits))

    def simulate(vman_values):
        nonlocal eval_idx, best_final

        vman_t = piecewise_constant_control(control_times, vman_values)

        def rhs(t, z):
            return hybrid_ode(t, z, vman_t, model, x_scaler, y_scaler)

        sol = solve_ivp(rhs, t_span, z0, t_eval=t_eval_points, method="RK45")

        if sol.success:
            final_biomass = sol.y[2, -1]

            store_full = (
                log_full_every_k is not None
                and log_full_every_k > 0
                and (eval_idx % log_full_every_k == 0)
            )

            is_new_best = final_biomass > best_final
            if is_new_best:
                best_final = final_biomass
                if log_best_full:
                    store_full = True

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

    def cost(vman_values):
        key = _cache_key(vman_values)
        if key in cache:
            return cache[key]

        sol = simulate(np.asarray(vman_values, dtype=float))
        if sol is None or not sol.success:
            val = np.inf
        else:
            val = -objective_fn(sol)

        cache[key] = val
        if verbose and np.isfinite(val):
            print(f"[eval {eval_idx-1}] biomass={-val:.6f} cost={val:.6f}")
        return val

    # If user doesn't pass an initial guess, use midpoints
    if initial_guess is None:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
        initial_guess = (lb + ub) / 2.0

    # --- Phase A: Global exploration (bounded) ---
    de_result = differential_evolution(
        cost,
        bounds=bounds,
        maxiter=global_maxiter,
        popsize=global_popsize,
        seed=seed,
        polish=False,          # polish later with Powell (multi-start)
        updating="deferred",
        disp=verbose,
    )

    # Collect candidate starting points for local polish
    starts = []

    # 1) Top-k individuals from the FINAL DE population
    #    (energies are the objective values; lower is better)
    pop = de_result.population
    energies = de_result.population_energies

    order = np.argsort(energies)  # ascending cost
    k = int(min(topk_polish, len(order)))

    starts.extend([pop[i] for i in order[:k]])

    # 2) Ensure the reported DE best is included (normally it is, but be safe)
    starts.append(de_result.x)

    # 3) Optionally include the user's initial guess as an additional start
    starts.append(np.asarray(initial_guess, dtype=float))

    def _unique_starts(starts_list, ndigits=6):
        """Deduplicate start points (avoids polishing near-identical candidates)."""
        seen = set()
        unique = []
        for s in starts_list:
            s = np.asarray(s, dtype=float)
            key = tuple(np.round(s, ndigits))
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    starts = _unique_starts(starts)

    # --- Phase B: Local polish (Powell) from multiple starts ---
    best_local = None

    for j, x0 in enumerate(starts):
        local_j = minimize(
            cost,
            x0=x0,
            bounds=bounds,
            method="Powell",
            options={"maxiter": polish_maxiter, "ftol": 1e-6},
        )

        if best_local is None or local_j.fun < best_local.fun:
            best_local = local_j

        if verbose:
            print(
                f"[polish {j+1}/{len(starts)}] "
                f"start_cost={cost(x0):.6g} "
                f"final_cost={local_j.fun:.6g} "
                f"success={local_j.success}"
            )

    # Return best polished result as the main result, plus logs and de_result
    return best_local, logs