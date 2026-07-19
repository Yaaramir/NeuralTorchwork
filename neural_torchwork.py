import nnfs
from nnfs.datasets import spiral_data
import torch
import torch.nn as nn
import torch.optim as optim

# General settings
nnfs.init()

# Model class for simple forward / backward passing etc
class Model(nn.Module):

    # Model creatin with an input and an output layer, a ReLU and a Softmax function. CCE used for loss calculation.
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = nn.Linear(in_features=2, out_features=512)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=512, out_features=3)
        self.softmax = nn.Softmax(dim=1)
        self.cce = nn.CrossEntropyLoss()

    # Creates an Adam optimizer
    def init_optimizer(self):
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7, weight_decay=0)

    # Forwards data to get final predictions, loss and accuracy
    def forward(self, X, y):
        x = self.dense1(X)
        x = self.relu(x)
        x = self.dense2(x)
        self.predictions = self.softmax(x)
        self.loss = self.cce(x, y)
        #self.acc = TODO        

# Create model
model = Model()
model.init_optimizer()

# Create dataset
X_raw, y_raw = spiral_data(samples=100, classes=3)
X = torch.tensor(X_raw)
y = torch.tensor(y_raw)

# TRAINING
for epoch in range(10001):
    # Forward pass
    model.forward(X, y)

    # Backward pass and parameter update
    model.optimizer.zero_grad()
    model.loss.backward()
    model.optimizer.step()

    # Print progression
    # TODO: accuracy
    if not epoch % 1000:
        print(f"epoch: {epoch}, " +
              f"data_loss: {model.loss:.3f}, ")

print(f"\n{model.loss:.3f} TRAINING LOSS")


# VALIDATION
X_test_raw, y_test_raw = spiral_data(samples=100, classes=3)
X_test = torch.tensor(X_test_raw)
y_test = torch.tensor(y_test_raw)

model.forward(X_test, y_test)
print(f"{model.loss:.3f} VALIDATION LOSS")