"""
Logistic Regression with PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_tutorial.utils import get_device, get_parameter_count


def test_logistic_regression(show_plots=False):
    """
    Main test function

    Args:
        show_plots (bool): Flag for plotting the training outcome
    """

    device = get_device()
    print(f"PyTorch {torch.__version__}, using {device} device")

    # Hyperparameters
    n_samples = 1000
    output_dim = 3  # Number of classes
    n_epochs = 60
    learning_rate = 0.001
    batch_size = 32

    # Generate a 2D dataset with scikit-learn
    inputs, targets = make_blobs(  # pylint: disable=unbalanced-tuple-unpacking
        n_samples=n_samples,
        n_features=2,  # x- and y-coordinates
        centers=output_dim,
        cluster_std=0.5,
        random_state=0,
    )

    # Convert dataset to PyTorch tensors and put them on GPU memory (if available)
    x_train = torch.from_numpy(inputs).float().to(device)
    y_train = torch.from_numpy(targets).int().to(device)

    # Create data loader for loading data as batches
    blobs_dataloader = DataLoader(
        list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
    )

    # Create a logistic regression model for the 2D dataset
    model = nn.Linear(in_features=2, out_features=output_dim).to(device)

    # Print model architecture
    print(model)

    # Compute and print parameter count
    n_params = get_parameter_count(model)
    print(f"Model has {n_params} trainable parameters")
    # Number of entries is 2 (x- and y-coordinates) + 1 (bias)
    assert n_params == 3 * output_dim

    # Use cross-entropy loss function.
    # Softmax is computed internally to convert outputs into probabilities
    criterion = nn.CrossEntropyLoss()

    # Use a vanilla mini-batch stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Set the model to training mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.train()

    # Number of samples
    n_samples = len(blobs_dataloader.dataset)

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(blobs_dataloader)

    # Train the model
    for epoch in range(n_epochs):
        # Total loss for epoch, divided by number of batches to obtain mean loss
        epoch_loss = 0

        # Number of correct predictions in an epoch, used to compute epoch accuracy
        n_correct = 0

        # For each batch of data
        for x_batch, y_batch in blobs_dataloader:
            # Forward pass
            y_pred = model(x_batch)

            # Compute loss value
            loss = criterion(y_pred, y_batch)

            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Accumulate data for epoch metrics: loss and number of correct predictions
                epoch_loss += loss.item()
                n_correct += (
                    (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()
                )

        # Compute epoch metrics
        mean_loss = epoch_loss / n_batches
        epoch_acc = n_correct / n_samples

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Mean loss: {mean_loss:.5f}. Accuracy: {epoch_acc * 100:.2f}%"
            )

    if show_plots:
        # Improve plots appearance
        sns.set_theme()

        _ = plot_decision_boundaries(
            model=model,
            x=x_train,
            y=y_train,
            title="Logistic Regression with PyTorch",
            device=device,
        )
        plt.show()


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
    x_mesh = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

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


# Standalone execution
if __name__ == "__main__":
    test_logistic_regression(show_plots=True)
