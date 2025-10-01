# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        20250930_ee32009_snn.py
 Description:
 Author:       Joshua Poole
 Created on:   20251001
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    - Python >= 3.11
    - <List required libraries>

==================
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


# Simple class for a simple example nn

class BadNuralNet(nn.Module):
    def __init__(self):
        super(BadNuralNet, self).__init__()
        self.fc1 = nn.Linear(4, 16) # Weights are shaped [16, 4]
                                                         # Biases are shaped [16]
        self.fc2 = nn.Linear(16, 3) # Weights are shaped [3, 16]
                                                         # Biases are shaped [3]
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example code for getting started with torch

tensor_ex = torch.tensor([1.0, 2.0, 3.0])
print(tensor_ex)

# Move tensor to GPU (if available)
if torch.cuda.is_available():
    tensor_ex = tensor_ex.to("cuda")
    print("On GPU:", tensor_ex)

# Gradient calcs are important for weights in nns
# torch does grads for you...

x_var = torch.tensor(3.0, requires_grad=True)
y_var = x_var**2 + 3*x_var - 1 # This is a lil polynomial
y_var.backward() # This computes dy/dx
print(x_var.grad) # In this case this should print 9.0 I think (since dy/dx = 2x + 3, at x=3)

# Using BadNuralNet to illustrate a simple nn implementation

model = BadNuralNet()
print(model)

# Dummy data: 100 samples, 4 features each
X = torch.rand(100, 4)
y = torch.randint(0, 3, (100,))  # labels between 0 and 2
criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)

# This is the training loop where the weights are created in the nn

for epoch in range(10):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass + optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")




