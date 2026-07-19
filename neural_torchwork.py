import nnfs
from nnfs.datasets import spiral_data
import torch

# General settings
nnfs.init()

# Create dataset
X_raw, y_raw = spiral_data(samples=100, classes=3)
X = torch.tensor(X_raw)
y = torch.tensor(y_raw)