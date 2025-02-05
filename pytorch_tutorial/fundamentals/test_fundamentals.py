"""
PyTorch fundamentals
"""

import math
import numpy as np
from sklearn.datasets import make_circles
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from pytorch_tutorial.utils import get_device

# Directory for downloaded files and saved model weights
DATA_DIR = MODEL_DIR = "./_output"


def test_tensor_manipulation():
    """Test tensors manipulation"""

    # Create a 1D tensor with predefined values
    x = torch.tensor([5.5, 3])
    assert x.shape == torch.Size([2])
    assert x.dtype == torch.float32
    assert x.device == torch.device(type="cpu")

    # Create a 2D tensor filled with random integers.
    # Values are generated uniformly between the low and high (excluded) bounds
    x = torch.randint(low=0, high=100, size=(5, 3))
    assert x.shape == torch.Size([5, 3])
    assert x.dtype == torch.int64
    assert x.device == torch.device(type="cpu")

    # Addition operator
    y1 = x + 2

    # Addition method, obtaining (logically) and identical result
    y2 = torch.add(x, 2)
    assert torch.equal(y1, y2)

    # Create a deep copy of a tensor (allocating new memory).
    # detach() removes its output from the computational graph (no gradient computation).
    # See below for details about gradients.
    # See also https://stackoverflow.com/a/62496418
    x_clone = x.detach().clone()

    # In-place addition: tensor is mutated
    x.add_(2)
    assert torch.equal(x, x_clone + 2)

    # NumPy-like indexing and slicing: update all values of second axis
    x[:, 1] = 0

    # PyTorch allows a tensor to be a view of an existing tensor.
    # View tensors share the same underlying data with their base tensor.
    # Example : reshaping a 2D tensor into a 1D tensor (a vector)
    x_view = x.view(15)
    assert x_view.shape == torch.Size([15])

    # The dimension identified by -1 is inferred from other dimensions
    assert x.view(-1, 5).shape == torch.Size([3, 5])
    assert x.view(
        -1,
    ).shape == torch.Size([15])

    # The reshape() function mimics the NumPy API.
    # Example: reshaping into a (3,5) tensor, creating a view if possible
    assert x.reshape(3, -1).shape == torch.Size([3, 5])

    # Number of values in the next arrays/tensors
    n_values = 5

    # Create a PyTorch tensor from a NumPy array
    n = np.ones(n_values)
    t = torch.from_numpy(n)
    assert t.shape == torch.Size([n_values])
    # Updating the array mutates the tensor
    np.add(n, 1, out=n)
    assert torch.equal(t, torch.tensor([2] * n_values))

    # Obtain a NumPy array from a PyTorch tensor
    t = torch.ones(n_values)
    n = t.numpy()
    assert n.shape == (n_values,)
    # Updating the tensor mutates the array
    t.add_(1)
    assert np.array_equal(n, np.array([2] * n_values))


def test_gpu_support():
    """Test GPU support"""

    device = get_device()
    print(f"PyTorch {torch.__version__}, using {device} device")

    # Create a 1D tensor (filled with the scalar value 1) on the memory of the initialized device
    _ = torch.ones(5, device=device)

    # Create a 2D tensor (filled with zeros) on CPU memory
    x_cpu = torch.zeros(2, 3)

    # Copy tensor to GPU memory (if available)
    x_device = x_cpu.to(device)

    # Create a copy of a GPU-based tensor in CPU memory
    _ = x_device.cpu()

    # Obtain a NumPy array from a GPU-based tensor
    _ = x_device.detach().cpu().numpy()


def test_autodiff():
    """Test autodifferentiation engine"""

    # Example 1: basic operations

    # Create scalar tensors with gradient computation activated.
    # (By default, operations are not tracked on user-created tensors)
    x = torch.tensor(1.0, requires_grad=True)
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    # Apply operations
    y = w * x + b
    assert y.requires_grad is True

    # Compute gradients of operations leading up to this tennsor
    y.backward()

    # Print the gradients
    assert x.grad == 2  # x.grad = dy/dx = w
    assert w.grad == 1  # w.grad = dy/dw = x
    assert b.grad == 1  # b.grad = dy/db

    # no_grad() avoids tracking operations history when gradients computation is not needed
    with torch.no_grad():
        y_no = w * x + b
        assert y_no.requires_grad is False

    # Example 2: a slighly more complex computational graph

    # Create two scalar tensors with gradient computation activated
    x1 = torch.tensor([2.0], requires_grad=True)
    x2 = torch.tensor([5.0], requires_grad=True)

    # y = f(x1,x2) = ln(x1) + x1.x2 - sin(x2)
    v1 = torch.log(x1)
    v2 = x1 * x2
    v3 = torch.sin(x2)
    v4 = v1 + v2
    y = v4 - v3

    # Compute gradients
    y.backward()

    # dy/dx1 = 1/x1 + x2 = 1/2 + 5
    assert x1.grad == 5.5
    # dy/dx2 = x1 - cos(x2) = 2 - cos(5) = 1.7163...
    assert x2.grad == 2 - torch.cos(torch.tensor(5))


def test_dataset_loading():
    """Test dataset loading"""

    # Number of samples in each batch
    batch_size = 32

    # Example 1: loading an integrated dataset

    # Download and construct the MNIST handwritten digits training dataset
    mnist = datasets.MNIST(
        root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True
    )

    # Fetch one data pair (read data from disk)
    image, label = mnist[0]
    # MNIST samples are bitmap images of shape (color_depth, height, width).
    # Color depth is 1 for grayscale images
    assert image.shape == torch.Size([1, 28, 28])
    # Image label is a scalar value
    assert isinstance(label, int)

    # Data loader (this provides queues and threads in a very simple way).
    mnist_dataloader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

    # Number of batches in a training epoch (= n_samples / batch_size, rounded up)
    n_batches = len(mnist_dataloader)
    assert n_batches == math.ceil(len(mnist) / batch_size)

    # Loop-based iteration is the most convenient way to train models on batched data
    for x_batch, y_batch in mnist_dataloader:
        # x_batch contains inputs for the current batch
        assert x_batch.shape == torch.Size([batch_size, 1, 28, 28])
        # y_batch contains targets for the current batch
        assert y_batch.shape == torch.Size([batch_size])

        # ... (Training code for the current batch should be written here)

    # Example 2: loading a scikit-learn dataset

    # Number of generated samples
    n_samples = 500

    # Generate 2D data (two concentric circles)
    inputs, targets = make_circles(n_samples=n_samples, noise=0.1, factor=0.3)
    assert inputs.shape == (n_samples, 2)
    assert targets.shape == (n_samples,)

    # Create tensor for inputs
    x_train = torch.from_numpy(inputs).float()
    assert x_train.shape == torch.Size([n_samples, 2])

    # Create tensor for targets (labels)
    # PyTorch loss functions expect float results of shape (batch_size, 1) instead of (batch_size,)
    # So we add a new axis and convert them to floats
    y_train = torch.from_numpy(targets[:, np.newaxis]).float()
    assert y_train.shape == torch.Size([n_samples, 1])

    # Load data as randomized batches for training
    circles_dataloader = DataLoader(
        list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
    )

    # Number of batches in a training epoch (= n_samples / batch_size, rounded up)
    n_batches = len(circles_dataloader)
    assert n_batches == math.ceil(n_samples / batch_size)

    # ... (Use dataloader as seen above)

    # Example 3: loading a custom dataset

    class CustomDataset(Dataset):
        """A custom Dataset class must implement three functions: __init__, __len__, and __getitem__"""

        def __init__(self):
            # Init internal state (file paths, etc)
            # ...
            pass

        def __len__(self):
            # Return the number of samples in the dataset
            # ...
            return 1

        def __getitem__(self, index):
            # Load, preprocess and return one data sample (inputs and label)
            # ...
            pass

    custom_dataset = CustomDataset()

    _ = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)

    # ... (Use dataloader for batched access to data)


def test_model_loading_and_saving():
    """Test model loading and saving"""

    device = get_device()

    # Download and load the pretrained model ResNet-18
    resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")

    # Optional: copy downloaded model to device memory for hardware acceleration.
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model
    resnet = resnet.to(device)

    # Save model parameters (recommended way of saving models)
    resnet_weights_filepath = f"{MODEL_DIR}/resnet_weights.pth"
    torch.save(resnet.state_dict(), resnet_weights_filepath)

    # Load untrained model ResNet-18 on device momory
    resnet = models.resnet18().to(device)

    # Load saved weights (results of the training process)
    resnet.load_state_dict(torch.load(resnet_weights_filepath, weights_only=True))

    # Set model to evaluation mode (needed for consistent inference results).
    # Model is now ready for inference
    resnet.eval()


# Standalone execution
if __name__ == "__main__":
    test_tensor_manipulation()
    test_gpu_support()
    test_autodiff()
    test_dataset_loading()
    test_model_loading_and_saving()
