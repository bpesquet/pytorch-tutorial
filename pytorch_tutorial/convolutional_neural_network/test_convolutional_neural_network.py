"""
Convolutional Neural Network (CNN) a.k.a. convnet
"""

import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_tutorial.utils import (
    get_device,
    get_parameter_count,
    plot_fashion_images,
)

# Directory for downloaded files
DATA_DIR = "./_output"


class Convnet(nn.Module):
    """Convnet for fashion articles classification"""

    def __init__(self, n_classes=10):
        super().__init__()

        # Define a sequential stack of layers
        self.layer_stack = nn.Sequential(
            # 2D convolution, output shape: (batch_zize, 32, 26, 26) with Fashion-MNIST images
            # Without padding, output_dim = (input_dim - kernel_size + 1) / stride
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            # Max pooling, output shape: (batch_zize, 32, 13, 13) with Fashion-MNIST images
            nn.MaxPool2d(kernel_size=2),
            # 2D convolution, output shape: (batch_zize, 64, 11, 11) with Fashion-MNIST images
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            # Max pooling, output shape: (batch_zize, 64, 5, 5) with Fashion-MNIST images
            nn.MaxPool2d(kernel_size=2),
            # Flattening layer, output shape: (batch_zize, 64x5x5 = 1600) with Fashion-MNIST images
            nn.Flatten(),
            # Linear layer whose input features are inferred during the first call to forward(). Output shape: (batch_zize, 128).
            # This avoids hardcoding the output shape of the previous layer, which depends on the shape of input images
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            # Output shape: (batch_size, 10)
            nn.Linear(in_features=128, out_features=n_classes),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Compute output of layer stack
        logits = self.layer_stack(x)

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits


def test_convolutional_neural_network(show_plots=False):
    """
    Main test function
    """

    device = get_device()
    print(f"PyTorch {torch.__version__}, using {device} device")

    # Hyperparameters
    n_epochs = 10
    learning_rate = 0.001
    batch_size = 64

    # Download and construct the Fashion-MNIST images dataset
    # The training set is used to train the model
    train_dataset = datasets.FashionMNIST(
        root=f"{DATA_DIR}",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    # The test set is used to evaluate the trained model performance on unseen data
    test_dataset = datasets.FashionMNIST(
        root=f"{DATA_DIR}",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Create data loader for loading training data as randomized batches
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    # Number of training samples
    n_train_samples = len(train_dataloader.dataset)

    # Number of batches in an epoch (= n_train_samples / batch_size, rounded up)
    n_batches = len(train_dataloader)
    assert n_batches == math.ceil(n_train_samples / batch_size)

    # Create data loader for loading test data as randomized batches
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    # Number of test samples
    n_test_samples = len(test_dataloader.dataset)

    print(f"{n_train_samples} training samples, {n_test_samples} test samples")

    # Create the convolutional network
    model = Convnet().to(device)

    # Use the first training image as dummy to initialize the LazyLinear layer.
    # This is mandatory to count model parameters (see below)
    first_img, _ = train_dataset[0]
    # Add a dimension (to match expected shape with batch size) and store tensor on device memory
    dummy_batch = first_img[None, :].to(device)
    model(dummy_batch)

    # Print model architecture
    print(model)

    # Compute and print parameter count
    n_params = get_parameter_count(model)
    print(f"Model has {n_params} trainable parameters")
    # Conv2d layers have (in_channels * kernel_size * kernel_size + 1) * out_channels parameters
    # Linear layers have (in_features + 1) * out_features parameters.
    # The following values must be changed if the model architecture is modified
    n_params_cond2d1 = (1 * 3 * 3 + 1) * 32
    n_params_cond2d2 = (32 * 3 * 3 + 1) * 64
    n_params_linear1 = (64 * 5 * 5 + 1) * 128
    n_params_linear2 = (128 + 1) * 10
    assert (
        n_params
        == n_params_cond2d1 + n_params_cond2d2 + n_params_linear1 + n_params_linear2
    )

    # Use cross-entropy loss function.
    # nn.CrossEntropyLoss computes softmax internally
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer for gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set the model to training mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.train()

    # Train the model
    for epoch in range(n_epochs):
        # Total loss for epoch, divided by number of batches to obtain mean loss
        epoch_loss = 0

        # Number of correct predictions in an epoch, used to compute epoch accuracy
        n_correct = 0

        for x_batch, y_batch in train_dataloader:
            # Copy batch data to GPU memory (if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

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
        epoch_acc = n_correct / n_train_samples

        print(
            f"Epoch [{(epoch + 1):3}/{n_epochs:3}] finished. Mean loss: {mean_loss:.5f}. Accuracy: {epoch_acc * 100:.2f}%"
        )

    # Set the model to evaluation mode - important for batch normalization and dropout layers.
    # Unnecessary here but added for best practices
    model.eval()

    # Compute model accuracy on test data
    with torch.no_grad():
        n_correct = 0

        for x_batch, y_batch in test_dataloader:
            # Copy batch data to GPU memory (if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = model(x_batch)
            n_correct += (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()

        test_acc = n_correct / len(test_dataloader.dataset)
        print(f"Test accuracy: {test_acc * 100:.2f}%")

    if show_plots:
        # Plot several test images and their associated predictions
        _ = plot_fashion_images(data=test_dataset, device=device, model=model)
        plt.show()


# Standalone execution
if __name__ == "__main__":
    test_convolutional_neural_network(show_plots=True)
