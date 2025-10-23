# plotting functions

import numpy as np
import matplotlib.pyplot as plt

def plot_flux_space(X, Y, vman_id="PFK", output_labels=None):
    """
    Plot exchange fluxes vs vman values.

    Parameters
    ----------
    X : list of [float]
        List of input control fluxes (vman values)
    Y : list of [float]
        Corresponding list of output flux vectors
    vman_id : str
        ID of the manipulated reaction (used for x-axis label)
    output_labels : list of str, optional
        Names of the output fluxes in Y (used for legend)
    """
    X_vals = np.array(X).flatten()
    Y_vals = np.array(Y)

    n_outputs = Y_vals.shape[1]

    # If not provided, create default labels
    if output_labels is None:
        output_labels = [f"Flux {i}" for i in range(n_outputs)]

    plt.figure(figsize=(8, 5))
    for i in range(n_outputs):
        plt.plot(X_vals, Y_vals[:, i], label=output_labels[i])

    plt.xlabel(f"{vman_id} value")
    plt.ylabel("Exchange fluxes")
    plt.title("Flux Space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    