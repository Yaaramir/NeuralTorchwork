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

    # Creates an Adam optimizer with learning rate configuration
    def init_optimizer(self):
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=0.05,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 1.0 / (1.0 + 5e-7 * epoch)
        )

    # Forwards data to get final predictions, loss and accuracy
    def forward(self, X, y):

        # pass data X forward
        logits = self.dense1(X)
        logits = self.relu(logits)
        logits = self.dense2(logits)

        # Evaluation: loss and accuracy
        self.loss = self.cce(logits, y)
        classes = torch.argmax(logits, dim=1)
        true_classes = torch.argmax(y, dim=1) if y.ndim == 2 else y
        self.acc = (classes == true_classes).float().mean()
        

# Create model
model = Model()
model.init_optimizer()

# TRAINING
# Create training dataset
X_raw, y_raw = spiral_data(samples=100, classes=3)
X = torch.tensor(X_raw)
y = torch.tensor(y_raw)

# Iterations
for epoch in range(10001):
    # Forward pass
    model.forward(X, y)

    # Backward pass and parameter update
    model.optimizer.zero_grad()
    model.loss.backward()
    model.optimizer.step()
    model.scheduler.step()

    # Print progression
    if not epoch % 1000:
        print(f"epoch: {epoch}, " +
              f"accuracy: {model.acc:.3f}, " +
              f"loss: {model.loss:.3f}, " +
              # TODO
              f"learning rate: {model.scheduler.get_last_lr()}")        
loss_train = model.loss
acc_train = model.acc

# VALIDATION
# Create validation dataset
X_val_raw, y_val_raw = spiral_data(samples=100, classes=3)
X_val = torch.tensor(X_val_raw)
y_val = torch.tensor(y_val_raw)

model.forward(X_val, y_val)
loss_val = model.loss
acc_val = model.acc

# EVALUATION
# Printing results
print(f"\n{loss_train:.3f} TRAINING LOSS")
print(f"{loss_val:.3f} VALIDATION LOSS ({(loss_val - loss_train):.3f})\n")

print(f"{acc_train:.3f} TRAINING ACCURACY")
print(f"{acc_val:.3f} VALIDATION ACCURACY ({(acc_val - acc_train):.3f})\n")