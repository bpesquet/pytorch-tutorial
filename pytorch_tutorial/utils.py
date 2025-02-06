"""
Utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def get_device():
    """Return GPU device if available, or fall back to CPU"""

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def get_parameter_count(model):
    """
    Return the number of trainable parameters for a PyTorch model

    Args:
        model (torch.nn.Module): a PyTorch model
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_results(model, x, y, title):
    """
    Plot data and model predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model
        x (torch.Tensor): Input features of shape (n_samples, 2)
        y (torch.Tensor): Labels of shape (n_samples,)
        title (str): Plot title
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.eval()

    # Compute model results on training data, and convert them to a NumPy array
    y_pred = model(x).detach().cpu().numpy()

    # Convert inputs and targets to NumPy arrays
    x_cpu = x.detach().cpu().numpy()
    y_cpu = y.detach().cpu().numpy()

    # Plot the training results
    plt.plot(x_cpu, y_cpu, "ro", label="Original data")
    plt.plot(x_cpu, y_pred, label="Fitted line")
    plt.legend()
    plt.title(title)

    return plt.gcf()


def plot_decision_boundaries(model, x, y, title, device):
    """
    Plot the decision boundaries and data points for a PyTorch classifier.

    Args:
        model (torch.nn.Module): Trained PyTorch model
        inputs (torch.Tensor): Input features of shape (n_samples, 2)
        targets (torch.Tensor): Labels of shape (n_samples,)
        title (str): Plot title
        device (torch.device): device where data on model are stored
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.eval()

    # Convert inputs and targets to NumPy arrays
    x_cpu = x.detach().cpu().numpy()
    y_cpu = y.detach().cpu().numpy()

    # Determine bounds for the grid
    x_min, x_max = x_cpu[:, 0].min() - 1, x_cpu[:, 0].max() + 1
    y_min, y_max = x_cpu[:, 1].min() - 1, x_cpu[:, 1].max() + 1

    # Generate a grid of points with distance h between them
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Convert mesh to PyTorch tensors and put it on device memory
    x_mesh = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float).to(device)

    # Get predictions for mesh points
    with torch.no_grad():
        y_mesh = model(x_mesh).detach().cpu()
        if y_mesh.shape[1] > 1:  # For multi-class problems
            y_mesh = torch.argmax(y_mesh, dim=1)
        else:  # For binary classification
            y_mesh = (y_mesh > 0).float()

    # Reshape predictions to match mesh shape
    y_mesh = y_mesh.numpy().reshape(xx.shape)

    # Create the plot
    plt.figure()

    # Plot decision boundaries
    plt.contourf(xx, yy, y_mesh, alpha=0.4, cmap="RdYlBu")
    plt.contour(xx, yy, y_mesh, colors="k", linewidths=0.5)

    # # Plot data points
    scatter = plt.scatter(
        x_cpu[:, 0], x_cpu[:, 1], c=y_cpu, cmap="RdYlBu", linewidth=1, alpha=0.8
    )

    # Add legend
    unique_labels = np.unique(y_cpu)
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scatter.cmap(scatter.norm(label.item())),
            markersize=10,
            label=f"Class {label.item()}",
        )
        for label in unique_labels
    ]
    plt.legend(handles=legend_elements)

    plt.title(title)

    return plt.gcf()
