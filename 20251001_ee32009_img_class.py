# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        20250930_ee32009_img_class.py
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

from model_nn import *

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random




def view_classification(image, probabilities):
    probabilities = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    image = image.permute(1, 2, 0)
    denormalized_image= image / 2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


# Load in the model or create and load it
try:
    model = ConvolutionNet()
    state_dict = torch.load("img_class_nn.pth")
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded")
except:
    train_mod()
    model = torch.load("img_class_nn.pth")
    model.eval()



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

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,shuffle=False)
images, _ = next(iter(test_loader))


# let's grab a random image and show the probability that the nn determines for each class
# load in a random image from the mf test set
# change this value from 0-3 to get see any of the four samples
image = images[2]
batched_image = image.unsqueeze(0).to(device)
with torch.no_grad():
    log_probabilities = model(batched_image)

probabilities = torch.exp(log_probabilities).squeeze().cpu()
view_classification(image, probabilities)

# Evaluate the performance of the model against know test data

correct = 0
total = 0

# during inference (predictions) torch does not need to track grads
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
