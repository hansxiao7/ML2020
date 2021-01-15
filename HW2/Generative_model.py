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

X_train_case_1 = []
Y_train_case_1 = []
X_train_case_2 = []
Y_train_case_2 = []
n_train_case = int(n_case*0.7)
for i in range(n_train_case):
    n_type = data[i, n_parameters]
    if n_type == 0:
        X_train_case_1.append(data[i, 0:n_parameters])
        Y_train_case_1.append(data[i, n_parameters])
    else:
        X_train_case_2.append(data[i, 0:n_parameters])
        Y_train_case_2.append(data[i, n_parameters])

X_val = []
Y_val = []
for j in range(n_train_case, n_case):
    X_val.append(data[j, 0:n_parameters])
    Y_val.append(data[j, n_parameters])

X_mean_case_1 = np.reshape(np.mean(X_train_case_1, axis=0), (n_parameters,1))
X_mean_case_2 = np.reshape(np.mean(X_train_case_2, axis=0), (n_parameters,1))
n_case_1 = len(X_train_case_1)
n_case_2 = len(X_train_case_2)

X_train_case_1 = np.reshape(X_train_case_1, (n_case_1, n_parameters))
X_train_case_2 = np.reshape(X_train_case_2, (n_case_2, n_parameters))

X_cov_case_1 = np.cov(X_train_case_1.T)
X_cov_case_2 = np.cov(X_train_case_2.T)

X_cov = X_cov_case_1 * n_case_1/(n_case_1 + n_case_2) + X_cov_case_2 * n_case_2/(n_case_1 + n_case_2)

D = n_parameters

inv = np.linalg.pinv(X_cov)
w = ((X_mean_case_1 - X_mean_case_2).T).dot(inv)
b = -0.5 * np.linalg.multi_dot([X_mean_case_1.T, inv, X_mean_case_1]) + 0.5 * np.linalg.multi_dot([X_mean_case_2.T, inv, X_mean_case_2]) + np.log(n_case_1/n_case_2)


train_loss = 0
for k in X_train_case_1:
    z = np.dot(w, k) + b
    f_case_1 = 1/ (1 + math.exp(-z))
    if f_case_1 <= 0.5:
        train_loss += 1
for m in X_train_case_2:
    z = np.dot(w, m) + b
    f_case_1 = 1 / (1 + math.exp(-z))
    if f_case_1 > 0.5:
        train_loss += 1

print('Train loss:', train_loss/(n_case_1 + n_case_2))

val_loss = 0
for n in range(len(X_val)):
    x = X_val[n]
    target = Y_val[n]
    z = np.dot(w, x) + b
    f_case_1 = 1 / (1 + math.exp(-z))
    if target == 0:
        if f_case_1 <= 0.5:
            val_loss += 1
    else:
        if f_case_1 > 0.5:
            val_loss += 1

print('Validation loss:', val_loss/len(Y_val))





