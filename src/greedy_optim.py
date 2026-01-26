import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from flux_config import STATE_INDEX


def _objective_state_index(objective: str) -> int:
    if objective not in STATE_INDEX:
        raise ValueError(f"Unknown objective '{objective}'. Valid options: {sorted(STATE_INDEX.keys())}")
    return STATE_INDEX[objective]


def objective_fn(sol, t_span, objective="biomass"):
    """Final-value objective (matches your current fallback behavior)."""
    obj_idx = _objective_state_index(objective)
    return float(sol.y[obj_idx, -1])


def make_piecewise_constant_control(boundaries, values):
    """
    boundaries: array length K+1 [t0, ..., t1] strictly increasing
    values:     array length K (one value per interval)
    """
    boundaries = np.asarray(boundaries, dtype=float)
    values = np.asarray(values, dtype=float)

    def control(t):
        idx = np.searchsorted(boundaries, t, side="right") - 1
        idx = int(np.clip(idx, 0, len(values) - 1))
        return values[idx]

    return control


def _simulate(
    model,
    hybrid_ode,
    z0,
    t_span,
    t_eval_points,
    boundaries,
    values,
    x_scaler,
    y_scaler,
    objective="biomass",
):
    vman_t = make_piecewise_constant_control(boundaries, values)

    def rhs(t, z):
        return hybrid_ode(t, z, vman_t, model, x_scaler, y_scaler)

    sol = solve_ivp(rhs, t_span, z0, t_eval=t_eval_points, method="RK45")
    if not sol.success:
        return sol, -np.inf

    score = objective_fn(sol, t_span, objective=objective)
    return sol, score


def _polish_values_fixed_knots(
    model,
    hybrid_ode,
    z0,
    t_span,
    t_eval_points,
    boundaries,
    values0,
    value_bounds,
    x_scaler,
    y_scaler,
    objective="biomass",
    maxiter=200,
):
    """
    Local polish of segment values only, keeping boundaries fixed.
    Uses Powell on K variables.
    """
    K = len(values0)
    bounds = [value_bounds] * K

    cache = {}

    def _key(x, nd=10):
        return tuple(np.round(np.asarray(x, dtype=float), nd))

    def cost(x):
        key = _key(x)
        if key in cache:
            return cache[key]
        sol, score = _simulate(
            model, hybrid_ode, z0, t_span, t_eval_points,
            boundaries, x, x_scaler, y_scaler, objective
        )
        val = np.inf if (sol is None or not sol.success) else -score
        cache[key] = val
        return val

    res = minimize(
        cost,
        x0=np.asarray(values0, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-6},
    )
    return res


def optimize_vman_greedy_single_cut(
    
    model,
    hybrid_ode,
    z0,
    t_span,
    t_eval_points,
    value_bounds,                 # (lower, upper)
    x_scaler,
    y_scaler,
    objective="biomass",
    split_grid_size=40,           # number of candidate split times
    min_dt=1e-3,                  # exclude splits too close to ends
    inner_polish_maxiter=80,      # optimize 2 values for each candidate split
    final_polish_maxiter=300,     # optimize all values after choosing split
    seed=0,
    verbose=False,
    log_full_every_k=10,
    log_best_full=True,
):
    """
    Greedy optimizer specialized to the "one hard cut" hypothesis:
      - choose best single split time tau
      - optimize (v_left, v_right) for that tau
      - optional final polish (still 2 vars, but re-run cleanly)
    Returns (best_result_dict, logs)
    """
    rng = np.random.default_rng(seed)
    t0, t1 = map(float, t_span)

    logs = []
    eval_idx = 0
    best_score = -np.inf
    best_pack = None

    # ----- Step 0: baseline (no split, 1 interval) -----
    v_mid = float(0.5 * (value_bounds[0] + value_bounds[1]))
    boundaries0 = np.array([t0, t1], dtype=float)
    values0 = np.array([v_mid], dtype=float)

    sol0, score0 = _simulate(
        model, hybrid_ode, z0, t_span, t_eval_points,
        boundaries0, values0, x_scaler, y_scaler, objective
    )

    best_score = score0 if sol0.success else -np.inf
    best_pack = {
        "boundaries": boundaries0,
        "values": values0,
        "tau": None,
        "score": float(best_score),
        "sol": sol0,
    }

    if verbose:
        print(f"[baseline] score={best_score:.6g} v={v_mid:.4g}")

    # candidate split times
    # exclude ends by min_dt
    taus = np.linspace(t0 + min_dt, t1 - min_dt, split_grid_size)

    # small helper: for a fixed tau, optimize 2 values
    def best_values_for_tau(tau, init=None):
        boundaries = np.array([t0, float(tau), t1], dtype=float)

        if init is None:
            # start near midpoint with small random perturbation (helps a bit)
            init = np.array(
                [v_mid, v_mid] + 0.05 * (value_bounds[1] - value_bounds[0]) * rng.standard_normal(2),
                dtype=float,
            )
            init = np.clip(init, value_bounds[0], value_bounds[1])

        res = _polish_values_fixed_knots(
            model=model,
            hybrid_ode=hybrid_ode,
            z0=z0,
            t_span=t_span,
            t_eval_points=t_eval_points,
            boundaries=boundaries,
            values0=init,
            value_bounds=value_bounds,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            objective=objective,
            maxiter=inner_polish_maxiter,
        )

        vals = np.asarray(res.x, dtype=float)
        sol, score = _simulate(
            model, hybrid_ode, z0, t_span, t_eval_points,
            boundaries, vals, x_scaler, y_scaler, objective
        )
        return boundaries, vals, sol, score, res

    # ----- Step 1: sweep taus -----
    for tau in taus:
        boundaries, vals, sol, score, inner_res = best_values_for_tau(tau)

        store_full = (
            log_full_every_k is not None
            and log_full_every_k > 0
            and (eval_idx % log_full_every_k == 0)
        )
        is_new_best = (sol.success and score > best_score)
        if is_new_best:
            best_score = float(score)
            if log_best_full:
                store_full = True

        entry = {
            "eval_idx": eval_idx,
            "tau": float(tau),
            "boundaries": np.copy(boundaries),
            "vman_values": np.copy(vals),
            "objective": objective,
            "score": float(score) if sol.success else -np.inf,
            "solver_success": bool(sol.success),
            "store_full": bool(store_full),
            "inner_fun": float(inner_res.fun),
            "inner_success": bool(inner_res.success),
        }
        if store_full and sol.success:
            obj_idx = _objective_state_index(objective)
            entry["t"] = sol.t
            entry["objective_curve"] = sol.y[obj_idx, :]

        logs.append(entry)

        if is_new_best:
            best_pack = {
                "boundaries": boundaries,
                "values": vals,
                "tau": float(tau),
                "score": float(score),
                "sol": sol,
            }
            if verbose:
                print(f"[best@eval{eval_idx}] tau={tau:.4g} score={score:.6g} v={vals}")

        eval_idx += 1

    # ----- Step 2: final polish at the best tau (optional but recommended) -----
    if best_pack["tau"] is not None and final_polish_maxiter is not None and final_polish_maxiter > 0:
        tau = best_pack["tau"]
        boundaries = best_pack["boundaries"]
        init = best_pack["values"]

        final_res = _polish_values_fixed_knots(
            model=model,
            hybrid_ode=hybrid_ode,
            z0=z0,
            t_span=t_span,
            t_eval_points=t_eval_points,
            boundaries=boundaries,
            values0=init,
            value_bounds=value_bounds,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            objective=objective,
            maxiter=final_polish_maxiter,
        )

        vals = np.asarray(final_res.x, dtype=float)
        sol, score = _simulate(
            model, hybrid_ode, z0, t_span, t_eval_points,
            boundaries, vals, x_scaler, y_scaler, objective
        )

        if sol.success and score >= best_pack["score"]:
            best_pack.update({"values": vals, "score": float(score), "sol": sol})
            if verbose:
                print(f"[final polish] tau={tau:.4g} score={score:.6g} v={vals}")

        best_pack["final_polish"] = {
            "success": bool(final_res.success),
            "fun": float(final_res.fun),
            "message": str(final_res.message),
            "nfev": getattr(final_res, "nfev", None),
            "nit": getattr(final_res, "nit", None),
        }

    # Return a clean dict without the raw sol object (you’ll re-simulate anyway)
    result = {
        "tau": best_pack["tau"],
        "boundaries": np.asarray(best_pack["boundaries"], dtype=float),
        "vman_values": np.asarray(best_pack["values"], dtype=float),
        "best_score": float(best_pack["score"]),
        "objective": objective,
    }
    return result, logs

def optimize_vman_greedy_k_cuts(
    model,
    hybrid_ode,
    z0,
    t_span,
    t_eval_points,
    value_bounds,
    x_scaler,
    y_scaler,
    objective="biomass",
    n_cuts=5,                    # number of cuts to insert (final intervals = n_cuts+1)
    split_grid_size=30,          # candidates per interval
    min_dt=1e-3,
    inner_polish_maxiter=80,
    polish_each_iter=False,      # if True, polish all values after each cut
    final_polish_maxiter=300,    # polish all values at end
    seed=0,
    verbose=False,
    log_full_every_k=10,
    log_best_full=True,
):
    rng = np.random.default_rng(seed)
    t0, t1 = map(float, t_span)

    logs = []
    eval_idx = 0
    best_score = -np.inf

    # Start with 1 interval
    boundaries = np.array([t0, t1], dtype=float)
    v_mid = float(0.5 * (value_bounds[0] + value_bounds[1]))
    values = np.array([v_mid], dtype=float)

    sol0, score0 = _simulate(
        model, hybrid_ode, z0, t_span, t_eval_points,
        boundaries, values, x_scaler, y_scaler, objective
    )
    if sol0.success:
        best_score = float(score0)

    def _log_entry(kind, **kwargs):
        nonlocal eval_idx
        entry = {"eval_idx": eval_idx, "kind": kind, "objective": objective}
        entry.update(kwargs)
        logs.append(entry)
        eval_idx += 1

    if verbose:
        print(f"[init] score={best_score:.6g} K={len(values)}")

    # helper: build a new (boundaries, values) by splitting interval j at tau
    def _apply_split(boundaries, values, j, tau, vL, vR):
        b_new = np.insert(boundaries, j + 1, float(tau))          # insert tau between b[j] and b[j+1]
        v_new = np.insert(values, j + 1, float(vR))               # add new right value
        v_new[j] = float(vL)                                      # replace old value with left value
        return b_new, v_new

    # helper: optimize vL,vR for a fixed (j,tau), keeping all other values fixed
    def _best_values_for_split(boundaries, values, j, tau):
        # initial guess around current value
        v0 = float(values[j])
        init = np.array(
            [v0, v0] + 0.05 * (value_bounds[1] - value_bounds[0]) * rng.standard_normal(2),
            dtype=float,
        )
        init = np.clip(init, value_bounds[0], value_bounds[1])

        def cost_pair(vpair):
            vpair = np.asarray(vpair, dtype=float)
            vL, vR = float(vpair[0]), float(vpair[1])
            b_new, v_new = _apply_split(boundaries, values, j, tau, vL, vR)
            sol, score = _simulate(
                model, hybrid_ode, z0, t_span, t_eval_points,
                b_new, v_new, x_scaler, y_scaler, objective
            )
            return np.inf if (sol is None or not sol.success) else -float(score)

        res = minimize(
            cost_pair,
            x0=init,
            method="Powell",
            bounds=[value_bounds, value_bounds],
            options={"maxiter": inner_polish_maxiter, "ftol": 1e-6},
        )

        vL, vR = float(res.x[0]), float(res.x[1])
        b_new, v_new = _apply_split(boundaries, values, j, tau, vL, vR)
        sol, score = _simulate(
            model, hybrid_ode, z0, t_span, t_eval_points,
            b_new, v_new, x_scaler, y_scaler, objective
        )
        return b_new, v_new, sol, float(score), res

    # Main loop: insert cuts
    for k in range(n_cuts):
        best_candidate = None  # (score, j, tau, b_new, v_new)

        # sweep each existing interval
        for j in range(len(values)):
            a, b = float(boundaries[j]), float(boundaries[j + 1])
            if (b - a) <= 2 * min_dt:
                continue

            taus = np.linspace(a + min_dt, b - min_dt, split_grid_size)

            for tau in taus:
                b_new, v_new, sol, score, inner_res = _best_values_for_split(boundaries, values, j, tau)

                store_full = (
                    log_full_every_k is not None
                    and log_full_every_k > 0
                    and (eval_idx % log_full_every_k == 0)
                )

                _log_entry(
                    "candidate",
                    iter=k,
                    interval_index=j,
                    tau=float(tau),
                    score=float(score) if sol.success else -np.inf,
                    solver_success=bool(sol.success),
                    inner_fun=float(inner_res.fun),
                    inner_success=bool(inner_res.success),
                    boundaries=np.copy(b_new),
                    vman_values=np.copy(v_new),
                    store_full=bool(store_full),
                )

                if store_full and sol.success:
                    obj_idx = _objective_state_index(objective)
                    logs[-1]["t"] = sol.t
                    logs[-1]["objective_curve"] = sol.y[obj_idx, :]

                if sol.success and (best_candidate is None or score > best_candidate[0]):
                    best_candidate = (score, j, float(tau), b_new, v_new)

        if best_candidate is None:
            if verbose:
                print(f"[iter {k+1}/{n_cuts}] no feasible split found; stopping.")
            break

        score, j_best, tau_best, b_best, v_best = best_candidate
        boundaries, values = b_best, v_best
        best_score = max(best_score, float(score))

        if verbose:
            print(f"[iter {k+1}/{n_cuts}] chose split in interval {j_best} at tau={tau_best:.4g} -> K={len(values)} score={score:.6g}")

        _log_entry(
            "accepted",
            iter=k,
            interval_index=j_best,
            tau=float(tau_best),
            score=float(score),
            boundaries=np.copy(boundaries),
            vman_values=np.copy(values),
            solver_success=True,
        )

        # optional polish of all values after each insertion
        if polish_each_iter and len(values) > 1:
            polish_res = _polish_values_fixed_knots(
                model=model,
                hybrid_ode=hybrid_ode,
                z0=z0,
                t_span=t_span,
                t_eval_points=t_eval_points,
                boundaries=boundaries,
                values0=values,
                value_bounds=value_bounds,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                objective=objective,
                maxiter=max(50, final_polish_maxiter // 3),
            )
            values = np.asarray(polish_res.x, dtype=float)

    # final polish of all values
    if final_polish_maxiter is not None and final_polish_maxiter > 0:
        polish_res = _polish_values_fixed_knots(
            model=model,
            hybrid_ode=hybrid_ode,
            z0=z0,
            t_span=t_span,
            t_eval_points=t_eval_points,
            boundaries=boundaries,
            values0=values,
            value_bounds=value_bounds,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            objective=objective,
            maxiter=final_polish_maxiter,
        )
        values = np.asarray(polish_res.x, dtype=float)

    # compute final score (for reporting)
    sol_final, score_final = _simulate(
        model, hybrid_ode, z0, t_span, t_eval_points,
        boundaries, values, x_scaler, y_scaler, objective
    )

    result = {
        "boundaries": np.asarray(boundaries, dtype=float),  # length K+1
        "vman_values": np.asarray(values, dtype=float),     # length K
        "n_cuts_requested": int(n_cuts),
        "n_cuts_used": int(len(values) - 1),
        "best_score": float(score_final) if sol_final.success else -np.inf,
        "objective": objective,
    }
    return result, logs