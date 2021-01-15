import numpy as np
import math

n_parameters = 510
n_case = 54256

X_train_raw = np.loadtxt('X_train', skiprows= 1, delimiter= ',', usecols=np.arange(1,n_parameters+1))
x_mean = np.mean(X_train_raw, axis=0)
x_std = np.std(X_train_raw, axis=0)
for i in range(n_case):
    X_train_raw[i, :] = (X_train_raw[i, :] - x_mean) / (x_std + 1e-8)

Y_train_raw = np.loadtxt('Y_train', skiprows= 1, delimiter= ',', usecols= 1)

Y_train_raw = np.reshape(Y_train_raw, (n_case, 1))

data = np.concatenate((X_train_raw, Y_train_raw), axis= 1)

np.random.shuffle(data)

X_train = []
Y_train = []
n_train_case = int(n_case*0.7)
for i in range(n_train_case):
    X_train.append(data[i, 0:n_parameters])
    Y_train.append(data[i, n_parameters])

X_val = []
Y_val = []
for j in range(n_train_case, n_case):
    X_val.append(data[j, 0:n_parameters])
    Y_val.append(data[j, n_parameters])
n_val_case = len(X_val)

#def learning rate and initial values
eta = 0.00001
w = np.zeros((n_parameters, 1))

b = 0

#sgd
n_iterations = 1000
for k in range(n_iterations):
    for m in range(n_train_case):
        x = np.reshape(X_train[m], (1, n_parameters))
        target = Y_train[m]
        z = np.dot(x, w) + b

        sigma_z = 1 / (1+math.exp(-z))
        b_gradient = - (target - sigma_z)
        w = w - eta * b_gradient * x.T
        b = b - eta * b_gradient

train_loss = 0
for n in range(n_train_case):
    x = X_train[n]
    target = Y_train[n]
    z = np.dot(x, w) + b
    sigma_z = 1 / (1 + math.exp(-z))
    if target == 1:
        if sigma_z <= 0.5:
            train_loss += 1
    else:
        if sigma_z > 0.5:
            train_loss += 1

print('Train loss:', train_loss/n_train_case)

val_loss = 0
for p in range(n_val_case):
    x = X_val[p]
    target = Y_val[p]
    z = np.dot(x, w) + b
    sigma_z = 1 / (1 + math.exp(-z))
    if target == 1:
        if sigma_z <= 0.5:
            val_loss += 1
    else:
        if sigma_z > 0.5:
            val_loss += 1

print('Validation loss:', val_loss/ n_val_case)





