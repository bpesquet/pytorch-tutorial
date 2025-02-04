"""
Linear Regression with PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn


def test_linear_regression(show_plots=False):
    """
    Main test function

    Args:
        show_plots (bool): Flag for plotting the training outcome
    """

    # Allow device detection code to be duplicated between examples
    # pylint: disable=duplicate-code

    # Accessing GPU device if available, or failing back to CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"PyTorch {torch.__version__}, using {device} device")

    # pylint: enable=duplicate-code

    # Hyperparameters
    input_dim = 1
    output_dim = 1
    n_epochs = 60
    learning_rate = 0.001

    # Toy dataset: inputs
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

    # Toy dataset: expected results
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

    # Convert dataset to PyTorch tensors and put them on GPU memory (if available)
    x_train = torch.from_numpy(inputs).to(device)
    y_train = torch.from_numpy(targets).to(device)

    # Create a Linear Regression model and put it on GPU memory
    model = nn.Linear(in_features=input_dim, out_features=output_dim).to(device)

    # Print model architecture and parameter count
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n_params} parameters")
    assert n_params == 2

    # Use Mean Squared Error loss
    criterion = nn.MSELoss()

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

        # Compute model results on training data, and convert them to a plain NumPy array
        predicted = model(x_train).detach().cpu().numpy()

        # Plot the training results
        plt.plot(inputs, targets, "ro", label="Original data")
        plt.plot(inputs, predicted, label="Fitted line")
        plt.legend()
        plt.title("Linear Regression with PyTorch")
        plt.show()


# Standalone execution
if __name__ == "__main__":
    test_linear_regression(show_plots=True)
