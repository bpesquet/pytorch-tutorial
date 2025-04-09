"""
Linear Regression with PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from pytorch_tutorial.utils import (
    get_device,
    get_parameter_count,
    plot_2d_data,
)


def test_linear_regression(show_plots=False):
    """
    Main test function

    Args:
        show_plots (bool): Flag for plotting the training outcome
    """

    device = get_device()
    print(f"PyTorch {torch.__version__}, using {device} device")

    # Hyperparameters
    n_epochs = 60  # Number of training iterations on the whole dataset
    learning_rate = 0.001  # Rate of parameter change during gradient descent

    # Toy dataset: inputs and expected results
    inputs = np.array(
        [
            [3.3],
            [4.4],
            [5.5],
            [6.71],
            [6.93],
            [4.168],
            [9.779],
            [6.182],
            [7.59],
            [2.167],
            [7.042],
            [10.791],
            [5.313],
            [7.997],
            [3.1],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [1.7],
            [2.76],
            [2.09],
            [3.19],
            [1.694],
            [1.573],
            [3.366],
            [2.596],
            [2.53],
            [1.221],
            [2.827],
            [3.465],
            [1.65],
            [2.904],
            [1.3],
        ],
        dtype=np.float32,
    )

    print(f"Inputs: {inputs.shape}. targets: {targets.shape}")

    # Convert dataset to PyTorch tensors and put them on GPU memory (if available)
    x_train = torch.from_numpy(inputs).to(device)
    y_train = torch.from_numpy(targets).to(device)

    # Create a Linear Regression model and put it on GPU memory
    model = nn.Linear(in_features=1, out_features=1).to(device)

    # Print model architecture
    print(model)

    # Compute and print parameter count
    n_params = get_parameter_count(model)
    print(f"Model has {n_params} trainable parameters")
    # Linear layers have (in_features + 1) * out_features parameters
    assert n_params == 2

    # Use Mean Squared Error loss
    criterion = nn.MSELoss()

    # Set the model to training mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.train()

    # Train the model
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(x_train)

        # Compute loss value
        loss = criterion(y_pred, y_train)

        # Reset the gradients to zero before running the backward pass.
        # Avoids accumulating gradients between GD steps
        model.zero_grad()

        # Compute gradients
        loss.backward()

        # no_grad() avoids tracking operations history when gradients computation is not needed
        with torch.no_grad():
            # Manual gradient descent step: update the weights in the opposite direction of their gradient
            for param in model.parameters():
                param -= learning_rate * param.grad

        # Print training progression
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Loss: {loss.item():.5f}"
            )

    if show_plots:
        # Improve plots appearance
        sns.set_theme()

        _ = plot_2d_data(
            x=x_train, y=y_train, model=model, title="Linear Regression with PyTorch"
        )
        plt.show()


# Standalone execution
if __name__ == "__main__":
    test_linear_regression(show_plots=True)
