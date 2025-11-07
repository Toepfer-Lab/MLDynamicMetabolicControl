import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def objective_fn(sol):
        print("in obj_fn!")
        print(f"biomass: {sol.y[2,-1]}")
        biomass_final = sol.y[2, -1] #assuming index 2 for biomass and y being time evo
        return biomass_final

def optimize_vman(model, hybrid_ode, z0, t_span, N, t_eval_points, bounds, x_scaler, y_scaler, log_trajectories=False):
    
    """
    Optimize the vman control trajectory to maximize biomass.
    Optionally logs every candidate trajectory and its biomass evolution.

    Parameters
    ----------
    model_nn : torch model
    hybrid_ode : callable
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
    log_trajectories : bool
        If True, stores each evaluated control trajectory and biomass evolution

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



    control_times = np.linspace(t_span[0], t_span[1], N+1) #split up the timespan into N+1 blocks
    logs=[]

    def simulate(vman_values):
        def vman_t(t):
            idx = np.searchsorted(control_times, t, side='right') - 1
            idx = np.clip(idx, 0 , N - 1)
            return vman_values[idx]
        

        def rhs(t, z):
            return hybrid_ode(t, z, vman_t, model, x_scaler, y_scaler)
            
        
        sol = solve_ivp(rhs, t_span, z0, t_eval=t_eval_points, method='RK45')
        
        if log_trajectories and sol.success:
             logs.append({
                  "vman_values": np.copy(vman_values),
                  "t": sol.t,
                  "biomass": sol.y[2,:],
                  "final_biomass": sol.y[2,-1],
             })

        return sol
    
    
    def cost(vman_values):
        sol = simulate(vman_values)
        print("in cost, should enter obj_fn now")
        if sol is None or not sol.success:
             return np.inf #penalizing integration failures
        return -objective_fn(sol)
    
    result = minimize(
        cost, 
        x0=np.ones(N)*5,
        bounds=bounds,
        method='L-BFGS-B',
    )
    
    if log_trajectories:
         return result, logs
    else:
        return result