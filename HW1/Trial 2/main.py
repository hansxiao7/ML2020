# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import transfer_train_file as train
train_file = 'train.csv'
train_data = train.train_data_generation(train_file)

w1_shape = (1, 18)
w2_shape = (9, 1)

#set initial values
w1 = np.random.randn(1, 18)
w2 = np.random.randn(9, 1)
b = 0
eta = 0.0005 #learning rate

n_iter = 100 #iteration time
train_loss = 0

for i in range(n_iter):
    for j in range(len(train_data)):
        x = train_data[j][0]
        y = np.dot(w1, x).dot(w2) + b
        target = train_data[j][1]

        w1_gradient = 2 * (y - target) * x.dot(w2).T
        w2_gradient = 2 * (y - target) * (w1.dot(x)).T
        b_gradient = 2 * (y - target)

        w1 = w1 - eta * w1_gradient
        w2 = w2 - eta * w2_gradient
        b = b - eta * b_gradient

        if i == n_iter - 1:
            train_loss += (target - y) ** 2

print(train_loss)