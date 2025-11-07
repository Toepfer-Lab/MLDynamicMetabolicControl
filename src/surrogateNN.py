import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class SurrogateNN(nn.Module):
    """
    A simple feedforward neural network (multi-layer perceptron)
    used as a surrogate model for steady-state FBA flux prediction.

    The model learns a mapping from a single manipulated intracellular flux
    (e.g., PFK) to several extracellular fluxes (e.g., ethanol, glucose, CO₂, biomass).

    Architecture:
        Input Layer  →  Hidden Layer (ReLU)  →  Output Layer

    Parameters
    ----------
    input_dim : int, optional (default=1)
        Number of input features (typically 1 for single control flux).
    hidden_dim : int, optional (default=5)
        Number of neurons in the hidden layer.
        In the referenced paper, 4-5 neurons were used.
    output_dim : int, optional (default=3)
        Number of output fluxes predicted (e.g., ethanol, CO₂, biomass).
    """

    def __init__(self, input_dim=1, hidden_dim=5, output_dim=3):
        super().__init__()

        # Define the network as a sequential stack of layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Linear layer: input → hidden
            nn.ReLU(),                         # Nonlinearity
            nn.Linear(hidden_dim, output_dim)  # Linear layer: hidden → output
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_dim].

        Returns
        -------
        torch.Tensor
            Predicted outputs of shape [batch_size, output_dim].
        """
        return self.net(x)

def ML_data_prep(X, Y, test_size=0.2, val_size=0.2):
    """
    Preprocess FBA data for machine learning:
    - Standardizes X and Y
    - Splits into train/val/test
    - Converts to PyTorch tensors

    Parameters
    ----------
    X : list or np.ndarray
        Input data (e.g., control fluxes)
    Y : list or np.ndarray
        Output data (e.g., external fluxes)
    test_size : float
        Fraction of data to use for testing (e.g. 0.2 = 20%)
    val_size : float
        Fraction of training data to use for validation

    Returns
    -------
    x_scaler, y_scaler : StandardScaler
        Fitted scalers for input/output
    X_train, Y_train, X_val, Y_val, X_test, Y_test : torch.Tensor
        Scaled and split training/validation/testing sets
    """
    X_np = np.array(X)
    Y_np = np.array(Y)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X_np)
    Y_scaled = y_scaler.fit_transform(Y_np)

    # Split: 20% test
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=test_size, random_state=42
    )

    # From remaining 80%, use 20% as validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=val_size, random_state=42
    )

    # Convert to PyTorch tensors
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    return (
        x_scaler, y_scaler,
        to_tensor(X_train), to_tensor(Y_train),
        to_tensor(X_val), to_tensor(Y_val),
        to_tensor(X_test), to_tensor(Y_test)
    )


def train_model(model, X_train, Y_train, X_val, Y_val,
                loss_fn=nn.MSELoss(), lr=1e-3, epochs=5000, patience=20, verbose=True):
    """
    Train a PyTorch model using early stopping based on validation loss.

    Parameters
    ----------
    model : nn.Module
        The neural network to train
    X_train, Y_train : torch.Tensor
        Training data
    X_val, Y_val : torch.Tensor
        Validation data
    loss_fn : nn.Module
        Loss function (default: MSE)
    lr : float
        Learning rate (default: 1e-3)
    epochs : int
        Maximum number of epochs (default: 5000)
    patience : int
        Early stopping patience (default: 20)
    verbose : bool
        If True, print progress every 100 epochs

    Returns
    -------
    model : nn.Module
        Trained model (with best weights restored)
    train_losses : list of float
        Training loss per epoch
    val_losses : list of float
        Validation loss per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training step
        model.train()
        pred_train = model(X_train)
        loss_train = loss_fn(pred_train, Y_train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())

        # Validation step
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val = loss_fn(pred_val, Y_val)
            val_losses.append(loss_val.item())

        # Logging
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f}")

        # Early stopping logic
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}    #create a clone of the best model state to avoid direct references to the model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch:03d} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f}")
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


