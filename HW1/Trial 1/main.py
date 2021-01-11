# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import transfer_train_file as train
train_file = 'train.csv'
train_data, [mean_x, std_x] = train.train_data_generation(train_file)

w_shape = (1, 18*9)

#get training data set and validation set, randomly chosen.
np.random.shuffle(train_data)
test_data = train_data[15:20]
train_data = train_data[0:15]

#set initial values
w = np.zeros(w_shape, dtype = float)
b = 0
eta = 0.0006 #learning rate

n_iter = 1000 #iteration time
train_loss = 0
test_loss = 0
result = []

for i in range(n_iter):
    train_loss = 0
    for j in range(len(train_data)):
        x = train_data[j][0]
        y = np.dot(w, x) + b
        target = train_data[j][1]

        w_gradient = 2 * (y - target) * x.T
        b_gradient = 2 * (y - target)

        w = w - eta * w_gradient
        b = b - eta * b_gradient

        train_loss += (target - y) ** 2

    result.append(train_loss)
np.savetxt('train_loss.txt', result)

for k in range(len(test_data)):

    x = test_data[k][0]
    y = np.dot(w, x) + b
    target = test_data[k][1]

    test_loss += (target - y) ** 2

print(test_loss/5)

x_train = []
y_train = []
x_test = []
y_test = []

for i in train_data:
    x_train.append(i[0])
    y_train.append(i[1])

for j in test_data:
    x_test.append(j[0])
    y_test.append(j[1])

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

LR = linear_model.LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
print(mean_squared_error(y_pred, y_test))