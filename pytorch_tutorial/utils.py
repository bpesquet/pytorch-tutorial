"""
Utilities
"""

import torch


def get_device():
    """Return GPU device if available, or fall back to CPU"""

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def get_parameter_count(model):
    """Return the number of trainable parameters for a PyTorch model"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
