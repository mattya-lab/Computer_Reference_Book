# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

def get_data(n):
    # Create sine waveforms
    wave_1 = 0.5 * np.sin(np.arange(0, n))
    wave_2 = 3.6 * np.sin(np.arange(0, n))
    wave_3 = 1.1 * np.sin(np.arange(0, n))
    wave_4 = 4.7 * np.sin(np.arange(0, n))
    # Create varying amplitudes
    amp_1 = np.ones(n)
    amp_2 = 2.1 + np.zeros(n)
    amp_3 = 3.2 * np.ones(n)
    amp_4 = 0.8 + np.zeros(n)
    
    w = np.array([wave_1, wave_2, wave_3, wave_4]).reshape(-1, 1)
    a = np.array([[amp_1, amp_2, amp_3, amp_4]]).reshape(-1, 1)
    return w, a

# Create some sample data
num_points = 40
wave, amp = get_data(num_points)
"""
plt.figure()
p1, = plt.plot(wave)
p2, = plt.plot(amp)
plt.legend([p1, p2], ['wave', 'amp'])
plt.show()
"""
# Create a recurrent neural network with 2 layers
nn = nl.net.newelm([[-2,2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
# Set the init functions for each layer
nn.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
nn.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
nn.init()
# Train the recurrent neural network 
error_progress = nn.train(wave, amp, epochs=1200, show=100, goal=0.01)
"""
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error(MSE)')
plt.show()
"""
#Visualize the output
def visualize_output(original, predicted, xlim=None):
    plt.figure()
    p1, = plt.plot(original)
    p2, = plt.plot(predicted)
    plt.legend([p1, p2], ['Original', 'Predicted'])
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()
# Run the training data thorugh the network    
output = nn.sim(wave)
#visualize_output(amp, output)
    
i, o = get_data(82)
p = nn.sim(i)
visualize_output(o, p, [0, 300])

i, o = get_data(30)
p = nn.sim(i)
visualize_output(o, p, [0, 300])