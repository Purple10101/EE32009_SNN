# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        model_nn.py
 Description:  src code for the model of the image
               classification training
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
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvolutionNet(nn.Module):
    def __init__(self):
        super(ConvolutionNet, self).__init__()

        # Convolution layers

        # conv1 takes 3 input features for RGB (red, green, blue)
        # We want to extract 64 features from each pixel
        # A kernal size of 3 indicates we are looking at 9 pixels at a time or a 3x3 chunk
        self.conv1 = nn.Conv2d(3, 64, 3)

        # conv2 takes 64 input features the conv1 output
        # We want to extract 128 features
        # A kernal size of 3 indicates we are looking at 9 chunks at a time
        self.conv2 = nn.Conv2d(64, 128, 3)

        # A max pooling layer that takes 2x2 window of the resulting feature map
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # This is like top level connection in HDL kinda.
        # You define thew interaction between the layers
        x = F.relu(self.conv1(x)) # relu is the activation function
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x # probability output for each class


def train_mod():

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(device)


    transform = transforms.Compose([
        transforms.ToTensor(), # Convert to tensors with value between 0-1
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ) # Standardize tensor value to be between 1 and -1
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    print('Number of images in the training dataset:', len(train_set))
    print(f"Shape of the images in the training dataset: {train_loader.dataset[0][0].shape}")
    # The shape is [3, 32, 32] 3 for RGB and 32x32 pixels

    # Plot the training data with known classes
    fig, axes = plt.subplots(1, 10, figsize=(12, 3))
    for i in range(10):
        image = train_loader.dataset[i][0].permute(1, 2, 0) # matplotlib needs the image data to be flipped
        denormalized_image= image / 2 + 0.5
        axes[i].imshow(denormalized_image)
        axes[i].set_title(classes[train_loader.dataset[i][1]])
        axes[i].axis('off')
    plt.show()


    img_nn = ConvolutionNet()
    img_nn.to(device)
    summary(img_nn, (3, 32, 32))

    # Now we have to train the model with a loss function

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(img_nn.parameters(), lr=0.001)

    epochs = 10 # iterate over the training data 10 times
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = img_nn(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(img_nn.state_dict(), "img_class_nn.pth")

if __name__ == '__main__':
    train_mod()