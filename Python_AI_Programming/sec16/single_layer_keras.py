# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10, activation='softmax')
 ])

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.5),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, batch_size=75, verbose=2)
score = model.evaluate(x_test, y_test, verbose=0)
#print('Accuracy =', score[1])

for i in range(10):
    plt.figure(figsize=(1, 1))
    score = model.predict(x_test[i].reshape(1, 28, 28))
    predicted = np.argmax(score)
    answer = np.argmax(y_test[i])
    
    plt.title('Answer:' + str(answer) +' Predicted:' + str(predicted))
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.show()