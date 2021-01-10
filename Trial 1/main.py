# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import transfer_train_file as train
train_file = 'train.csv'
train_data = train.train_data_generation(train_file)

w_shape = (1, 18*9)

#set initial values
w = np.zeros(w_shape, dtype = float)
b = 0
eta = 0.0005 #learning rate

n_iter = 100 #iteration time
train_loss = 0

for i in range(n_iter):
    for j in range(1):#len(train_data)):
        x = train_data[j][0]
        y = np.dot(w, x) + b
        target = train_data[j][1]

        w_gradient = 2 * (y - target) * x.T
        b_gradient = 2 * (y - target)

        w = w - eta * w_gradient
        b = b - eta * b_gradient

        if i == n_iter - 1:
            train_loss += (target - y) ** 2

print(train_loss)