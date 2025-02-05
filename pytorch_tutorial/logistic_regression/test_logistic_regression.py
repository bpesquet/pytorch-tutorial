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
from pytorch_tutorial.utils import get_device


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
    input_dim = 2
    output_dim = 3  # Number of classes
    n_epochs = 60
    learning_rate = 0.001
    batch_size = 32

    # Generate a toy dataset with scikit-learn
    inputs, targets = make_blobs(  # pylint: disable=unbalanced-tuple-unpacking
        n_samples=n_samples,
        n_features=input_dim,
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

    # Number of samples
    n_samples = len(blobs_dataloader.dataset)

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(blobs_dataloader)

    # Create a Logistic regression model
    model = nn.Linear(in_features=input_dim, out_features=output_dim).to(device)

    # Print model architecture and parameter count
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params} parameters")
    assert n_params == (input_dim + 1) * output_dim

    # Use cross-entropy loss function.
    # nn.CrossEntropyLoss computes softmax internally to convert model outputs into probabilities
    criterion = nn.CrossEntropyLoss()

    # Use a vanilla mini-batch stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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

        fig = plot_decision_boundary(
            model=model,
            inputs=x_train,
            targets=y_train,
            device=device,
            title="Logistic Regression with PyTorch",
        )
        plt.show()


def plot_decision_boundary(model, inputs, targets, device, title):
    """
    Plot the decision boundaries and data points for a PyTorch classifier.

    Args:
        model (torch.nn.Module): Trained PyTorch model
        inputs (torch.Tensor): Input features of shape (n_samples, 2)
        targets (torch.Tensor): Labels of shape (n_samples,)
        title (str): Plot title
    """
    # Set model to evaluation mode
    model.eval()

    inputs = inputs.detach().cpu()
    targets = targets.detach().cpu()

    # Determine bounds for the grid
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Convert mesh to PyTorch tensors
    X_mesh = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

    # Get predictions for mesh points
    with torch.no_grad():
        Z = model(X_mesh).detach().cpu()
        if Z.shape[1] > 1:  # For multi-class problems
            Z = torch.argmax(Z, dim=1)
        else:  # For binary classification
            Z = (Z > 0).float()

    # Reshape predictions to match mesh shape
    Z = Z.numpy().reshape(xx.shape)

    # Create the plot
    plt.figure()

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
    plt.contour(xx, yy, Z, colors="k", linewidths=0.5)

    # Plot data points
    scatter = plt.scatter(
        inputs[:, 0], inputs[:, 1], c=targets, cmap="RdYlBu", linewidth=1, alpha=0.8
    )

    # Customize plot
    plt.title(title)
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.colorbar(scatter)

    # Add legend
    unique_labels = torch.unique(targets)
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

    plt.tight_layout()
    return plt.gcf()


# Standalone execution
if __name__ == "__main__":
    test_logistic_regression(show_plots=True)
