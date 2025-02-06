"""
Logistic Regression with PyTorch
"""

import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_tutorial.utils import (
    get_device,
    get_parameter_count,
    plot_decision_boundaries,
)


def test_logistic_regression(show_plots=False):
    """
    Main test function

    Args:
        show_plots (bool): Flag for plotting the training outcome
    """

    device = get_device()
    print(f"PyTorch {torch.__version__}, using {device} device")

    # Hyperparameters
    n_samples = 1000  # Number of data samples
    output_dim = 3  # Number of classes
    n_epochs = 60  # Number of training iterations on the whole dataset
    learning_rate = 0.001  # Rate of parameter change during gradient descent
    batch_size = 32  # Number of samples used for one gradient descent step

    # Generate a 2D dataset with scikit-learn
    inputs, targets = make_blobs(  # pylint: disable=unbalanced-tuple-unpacking
        n_samples=n_samples,
        n_features=2,  # x- and y-coordinates
        centers=output_dim,
        cluster_std=0.5,
        random_state=0,
    )
    print(f"Inputs: {inputs.shape}. targets: {targets.shape}")
    assert inputs.shape == (n_samples, 2)
    assert targets.shape == (n_samples,)

    # Convert dataset to PyTorch tensors and put them on GPU memory (if available)
    x_train = torch.from_numpy(inputs).float().to(device)
    y_train = torch.from_numpy(targets).long().to(device)

    # Create data loader for loading data as randomized batches
    blobs_dataloader = DataLoader(
        list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
    )

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(blobs_dataloader)
    assert n_batches == math.ceil(n_samples / batch_size)

    # Create a logistic regression model for the 2D dataset
    model = nn.Linear(in_features=2, out_features=output_dim).to(device)

    # Print model architecture
    print(model)

    # Compute and print parameter count
    n_params = get_parameter_count(model)
    print(f"Model has {n_params} trainable parameters")
    # Linear layers have (in_features + 1) * out_features parameters
    assert n_params == 3 * output_dim

    # Use cross-entropy loss function.
    # Softmax is computed internally to convert outputs into probabilities
    criterion = nn.CrossEntropyLoss()

    # Use a vanilla mini-batch stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Set the model to training mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.train()

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


# Standalone execution
if __name__ == "__main__":
    test_logistic_regression(show_plots=True)
