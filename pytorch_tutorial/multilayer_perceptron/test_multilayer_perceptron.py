"""
MultiLayer Perceptron (MLP) a.k.a. Feedforward Neural Network 
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_tutorial.utils import (
    get_device,
    get_parameter_count,
    plot_decision_boundaries,
)


def test_multilayer_perceptron(show_plots=False):
    """
    Main test function.

    Args:
        show_plots (bool): Flag for plotting the training outcome
    """

    device = get_device()
    print(f"PyTorch {torch.__version__}, using {device} device")

    # Hyperparameters
    n_samples = 500  # Number of data samples
    hidden_layer_dim = 2  # Number of neurons on the hidden layer of the MLP
    n_epochs = 50  # Number of training iterations on the whole dataset
    learning_rate = 0.1  # Rate of parameter change during gradient descent
    batch_size = 5  # Number of samples used for one gradient descent step

    # Generate 2D data (a large circle containing a smaller circle)
    inputs, targets = make_circles(n_samples=n_samples, noise=0.1, factor=0.3)
    print(f"Inputs: {inputs.shape}. targets: {targets.shape}")
    assert inputs.shape == (n_samples, 2)
    assert targets.shape == (n_samples,)

    # Convert inputs to a PyTorch tensor and put it on GPU memory (if available)
    x_train = torch.from_numpy(inputs).float().to(device)
    assert x_train.shape == torch.Size([n_samples, 2])

    # Convert targets to a PyTorch tensor and put it on GPU memory (if available).
    # PyTorch loss function expects float results of shape (batch_size, 1) instead of (batch_size,).
    # So we add a new axis and convert them to floats
    y_train = torch.from_numpy(targets[:, np.newaxis]).float().to(device)
    assert y_train.shape == torch.Size([n_samples, 1])

    # Create data loader for loading data as randomized batches
    circles_dataloader = DataLoader(
        list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
    )

    # Number of batches in an epoch (= n_samples / batch_size, rounded up)
    n_batches = len(circles_dataloader)
    assert n_batches == math.ceil(n_samples / batch_size)

    # Create a MultiLayer Perceptron with 2 inputs, a hidden layer and 1 output
    model = nn.Sequential(
        # Hidden layer
        nn.Linear(in_features=2, out_features=hidden_layer_dim),
        # Activation function for the hidden layer
        nn.Tanh(),
        # Output layer
        nn.Linear(in_features=hidden_layer_dim, out_features=1),
        # Activation function for the output layer
        nn.Sigmoid(),
    ).to(device)

    # Print model architecture
    print(model)

    # Compute and print parameter count
    n_params = get_parameter_count(model)
    print(f"Model has {n_params} trainable parameters")
    # Hidden layer has (2 + 1) * hidden_layer_dim parameters.
    # Output layer has (hidden_layer_dim + 1) * 1 parameters
    assert n_params == 3 * hidden_layer_dim + hidden_layer_dim + 1

    # Use binary cross-entropy loss function
    criterion = nn.BCELoss()

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
        for x_batch, y_batch in circles_dataloader:
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
                    (torch.round(model(x_batch)) == y_batch).float().sum().item()
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
            title=f"MultiLayer Perceptron with PyTorch. Hidden layer dimension: {hidden_layer_dim}",
            device=device,
        )
        plt.show()


# Standalone execution
if __name__ == "__main__":
    test_multilayer_perceptron(show_plots=True)
