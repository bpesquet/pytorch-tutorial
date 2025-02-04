---
marp: true
math: true  # Use default Marp engine for math rendering
---

<!-- Apply header and footer to first slide only -->
<!-- _header: "[![Bordeaux INP logo](../ensc_logo.jpg)](https://www.bordeaux-inp.fr)" -->
<!-- _footer: "[Baptiste Pesquet](https://www.bpesquet.fr)" -->
<!-- headingDivider: 3 -->

# Linear Regression with PyTorch

<!-- Show pagination, starting with second slide -->
<!-- paginate: true -->

## Scope and objective

This [example](test_linear_regression.py) trains a Linear Regression model on a minimalist 2D dataset.

![Training outcome](images/linear_regression.png)

## Imports

First of all, we need to import the necessary stuff.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
```

## GPU support

Let's probe for the availability of an accelerated device.

```python
# Accessing GPU device if available, or failing back to CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"PyTorch {torch.__version__}, using {device} device")
```

## Hyperparameters

Next, we define the various hyperparameters for this example.

```python
# Hyperparameters
input_dim = 1
output_dim = 1
n_epochs = 60
learning_rate = 0.001
```

## Dataset loading

To keep things as simple as possible, the dataset is created from scratch as two NumPy arrays: `ìnputs` (x-coordinates of the samples) and `targets` (corresponding y-coordinates of the samples).

### Inputs

```python
# Toy dataset: inputs
inputs = np.array(
[
    [3.3],  [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182],
    [7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [3.1],
],
dtype=np.float32,
)
```

### Targets

```python
# Toy dataset: expected results
targets = np.array(
[
    [1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596],
    [2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [1.3],
],
dtype=np.float32,
)
```

### Tensors creation

Both inputs and targets are subsequently converted to PyTorch tensors stored into the device memory.

```python
# Convert dataset to PyTorch tensors and put them on GPU memory (if available)
x_train = torch.from_numpy(inputs).to(device)
y_train = torch.from_numpy(targets).to(device)
```

## Model definition

The Linear Regression model is implemented with the PyTorch [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) class, which applies an affine tranformation to its input.

```python
# Create a Linear Regression model and put it on GPU memory
model = nn.Linear(in_features=input_dim, out_features=output_dim).to(device)
```

This model defines a function $f(x) = w_0 + w_1 x$. It has two parameters: $w_0$ and $w_1$.

```python
# Print model architecture and parameter count
print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{n_params} parameters")
assert n_params == 2
```

## Loss function

The [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) class implements the Mean Squared Error loss function, well suited to regression tasks.

```python
# Use Mean Squared Error loss
criterion = nn.MSELoss()
```

## Training loop

The loop for training a PyTorch model in a supervised way has four main parts:

1. compute the outputs for a set of inputs;
2. compute the value of the loss function (difference between expected and actual values) for this set of inputs;
3. use autodiff to obtain the gradients of the loss functions w.r.t each model parameters;
4. update each parameter in the opposite direction of its gradient.

---

In this example, it is implemented in the simplest way possible.

- No batching: due to the small sample count, the whole dataset is used at each epoch (training iteration).
- Model parameters are updated by hand rather than by using a pre-built optimizer. This choice is made to better illustrate the Gradient Descent algorithm.

---

```python
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
```

## Results plotting

Finally model predictions (fitted line) are plotted alongside training data.

```python
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
```
