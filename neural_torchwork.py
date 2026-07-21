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
        # pass
        y_hat = self.dense1(X)
        y_hat = self.relu(y_hat)
        y_hat = self.dense2(y_hat)
        y_hat = self.softmax(y_hat)

        # evaluation: loss and accuracy
        self.loss = self.cce(y_hat, y)
        predictions = torch.argmax(y_hat, dim=1)
        y_true = torch.argmax(y, dim=1) if y.ndim == 2 else y
        self.acc = (predictions == y_true).float().mean()
        

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
    if not epoch % 1000:
        print(f"epoch: {epoch}, " +
              f"data_loss: {model.loss:.3f}, " +
              f"accuracy: {model.acc:.3f}")
        
loss_train = model.loss
acc_train = model.acc

# VALIDATION
X_val_raw, y_val_raw = spiral_data(samples=100, classes=3)
X_val = torch.tensor(X_val_raw)
y_val = torch.tensor(y_val_raw)

model.forward(X_val, y_val)
loss_val = model.loss
acc_val = model.acc

# Printing results
print(f"\n{loss_train:.3f} TRAINING LOSS")
print(f"{loss_val:.3f} VALIDATION LOSS ({(loss_val - loss_train):.3f})\n")

print(f"{acc_train:.3f} TRAINING ACCURACY")
print(f"{acc_val:.3f} VALIDATION ACCURACY ({(acc_val - acc_train):.3f})")