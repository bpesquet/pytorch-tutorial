"""
Logistic Regression with PyTorch
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.base import BaseEstimator
from sklearn.inspection import DecisionBoundaryDisplay
import torch
from torch import nn
from torch.utils.data import DataLoader


def test_logistic_regression(show_plots=False):
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

    if show_plots:
        # Improve plots appearance
        sns.set_theme()

        # Plot dataset
        plt.scatter(inputs[:, 0], inputs[:, 1], marker="o", c=targets, edgecolor="k")
        plt.title("Logistic Regression with PyTorch: dataset")
        plt.show()

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

        class Estimator(BaseEstimator):
            def __init__(self, model):
                self.model = model

            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.requires_fit = False
                return tags

            def fit(self, __):
                pass

            def predict(self, x):
                x_gpu = torch.from_numpy(x).float().to(device)
                return self.model(x_gpu).detach().cpu().numpy()

        classifier = Estimator(model=model)

        # Show datasets and classification boundaries
        plt.figure()
        disp = DecisionBoundaryDisplay.from_estimator(
            classifier,
            inputs,
            response_method="predict",
            alpha=0.5,
        )
        disp.ax_.scatter(inputs[:, 0], inputs[:, 1], c=targets, edgecolor="k")
        plt.title("Logistic Regression with PyTorch: training outcome")
        plt.show()


# Standalone execution
if __name__ == "__main__":
    test_logistic_regression(show_plots=True)
