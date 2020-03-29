# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Generate some training data
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
# y = 3*x^2 + 5
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

# Create data and labels
data = x.reshape(-1, 1)
labels = y.reshape(-1, 1)
"""
# Plot input data
plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')
"""
# Define a multilayer neural network with 2 hidden layers;
# First hidden layer consists of 10 neurons
# Second hidden layer consists of 6 neurons
# Output layer sonsists of 1 neuron
nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])
# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd
# Train the neural netwotk
error_progress = nn.train(data, labels, epochs=200, show=100, goal=0.01)
"""
# Plot training error
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
plt.show()
"""
# Run the neural network on training datapoints
output = nn.sim(data)
# Plot training error
plt.figure()
plt.scatter(data, labels, marker='.')
plt.scatter(data, output)
plt.title('Actual vs predicted')
plt.show()