from exceptiongroup import catch
import numpy as np
import matplotlib.pyplot as plt

import sys

if 'ipykernel' in sys.modules:
    # Running in Jupyter Notebook
    try:
        from flux_config import FLUX_LABELS
        print("1")
    except ImportError:
        print("2")
        from src.flux_config import FLUX_LABELS
else:
    # Running as a standard Python script
    try:
        print("3")
        from flux_config import FLUX_LABELS
    except ImportError:
        from src.flux_config import FLUX_LABELS


def plot_flux_space(X, Y, feasible_range, vman_id="ACKr", output_labels=None):
    """
    Plot exchange fluxes vs vman values with a secondary axis for biomass.
    """
    X_vals = np.array(X).flatten()
    #print(X_vals)
    Y_vals = np.abs(np.array(Y))

    n_outputs = Y_vals.shape[1]

    if output_labels is None:
        output_labels = list(FLUX_LABELS)

    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()

    for i in range(n_outputs):
        if i == 3:
            # Outlier on right axis
            ax_right.plot(
                X_vals,
                Y_vals[:, i],
                linestyle="--",
                label=output_labels[i]
            )
        else:
            # All others on left axis
            ax_left.plot(
                X_vals,
                Y_vals[:, i],
                label=output_labels[i]
            )

    # Axis labels
    ax_left.set_xlabel(f"$V_{{{vman_id}}}$ value")
    ax_left.set_ylabel(r"$V_{{ext},i}$ $[mmol/g_b/h]$")
    ax_right.set_ylabel(r"$V_{{ext},bio}$ $[1/h]$")

    ax_left.set_title("Flux Space")

    # Combine legends from both axes
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_right.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        borderaxespad=1,
    )

    ax_left.grid(True)
    fig.tight_layout()
    plt.show()

    #save the plot to /plots/flux_sweeps/flux_space.png
    fig.savefig("/home/jkaatz/MA/MLDynamicMetabolicControl/plots/flux_sweeps/flux_space.png", bbox_inches="tight")
