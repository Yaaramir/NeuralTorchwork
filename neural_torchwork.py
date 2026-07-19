import nnfs
from nnfs.datasets import spiral_data
import torch

# General settings
nnfs.init()

# Create model
dense1 = torch.nn.Linear(in_features=2, out_features=512)
activation1 = torch.nn.ReLU()
dense2 = torch.nn.Linear(in_features=512, out_features=3)
activation2 = torch.nn.Softmax()
#loss = TODO
#optimizer = TODO

# Create dataset
X_raw, y_raw = spiral_data(samples=100, classes=3)
X = torch.tensor(X_raw)
y = torch.tensor(y_raw)