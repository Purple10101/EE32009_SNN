# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        20250930_ee32009_perceptron_lab.py
 Description:
 Author:       Joshua Poole
 Created on:   20250930
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    - Python >= 3.11
    - <List required libraries>

==================
"""

import numpy as np
import matplotlib.pyplot as plt


# A single perceptron function
def perceptron(inputs_list, weights_list, bias) :
    # Convert the inputs list into a numpy array
    inputs = np.array(inputs_list)
    # Convert the weights list into a numpy array
    weights = np.array(weights_list)
    # Calculate the dot product
    summed = np.dot(inputs, weights)
    # Add in the bias
    summed = summed + bias
    # Calculate output
    # N.B this is a ternary operator , neat huh?
    output = 1 if summed > 0 else 0
    return output


# Exercises

# Sim for all possible binary combos of inputs
# and what logic gate does this represent
weights = [1.0, 1.0]
bias = -1

bits = 2
and_bool = True

res = []
inputs = []
for i in range(2**bits):
    binary = bin(i)[2:].zfill(2)
    inputs.append([float(binary[0]), float(binary[1])])
    # inputs[i] = [0.0, 0.0] for i=0
    # inputs[i] = [1.0, 1.0] for i=3

    res.append(perceptron(inputs[i], weights, bias))
    and_res = inputs[i][0] and inputs[i][1]

    # Check if the neuron and an AND gate still yield the same output
    if res[i] != and_res:
        and_bool = False

    print(" Inputs :", inputs[i])
    print(" Weights :", weights)
    print(" Bias :", bias)
    print(" Result :", res[i])
    print(" And parity :", and_bool)




# Compute decision boundary line
x_vals = np.linspace(-2, 2, 100)
y_vals = -(weights[0]/weights[1]) * x_vals - (bias/weights[1])

# plotting
for i in range(2**bits):
    plt.scatter(inputs[i][0], inputs[i][1], s=50, color="green" if res[i] else "red", zorder=3)

# Plot decision boundary
plt.plot(x_vals, y_vals, color="blue", linewidth=2, label="Decision boundary")

plt.xlim( -2 , 2)
plt.ylim( -2 , 2)
plt.xlabel(" Input 1")
plt.ylabel(" Input 2")
plt.title(" State Space of Input Vector ")
plt.grid( True , linewidth =1 , linestyle =":")
plt.tight_layout()
plt.show()
