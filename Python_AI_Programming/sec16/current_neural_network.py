# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import SGD

num_points = 1200
m = 0.2
c = 0.5
x_data = np.random.normal(0.0, 0.8, num_points)
noise = np.random.normal(0.0, 0.04, num_points)
y_data = m*x_data + c + noise
"""
plt.figure()
plt.plot(x_data, y_data, 'ro')
plt.title('Input data')
plt.show()
"""
num_iterations = 10

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        w = self.model.get_weights()[0][0][0]
        b = self.model.get_weights()[1][0]
        print('ITERATION', epoch+1)
        print('W =', w)
        print('b =', b)
        print('loss = ', logs.get('loss'))
        
        plt.figure()
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, w * x_data + b)
        plt.title('Iteration ' + str(epoch+1) + ' of ' + str(num_iterations)) 
        plt.show()
       

model = Sequential([
    Dense(1, activation='linear', input_shape=(1,),
          kernel_initializer=RandomUniform(-1.0, 1.0))
])
model.compile(loss='mse', optimizer=SGD(0.001))

history = model.fit(x_data, y_data, batch_size=1, epochs=num_iterations,
                    verbose=0, callbacks=[MyCallback()])       