import numpy as np
from collections import Counter

def train_data_generation(filename):
    train_file = open(filename, encoding= 'iso 8859-15')
    train_data_raw = np.loadtxt(train_file, delimiter= ',', dtype = str, skiprows = 1)

    n_parameters = 18
    n_date = len(Counter(train_data_raw[:, 0]).keys())

    train_data_raw[train_data_raw == 'NR'] = 0

    train_data_raw = train_data_raw[:, 3:].astype(np.float64)

    train_data = train_data_raw[:n_parameters, :]

    for i in range(1, n_date):
        temp = train_data_raw[(n_parameters * (i - 1)): (n_parameters * i), :]
        train_data = np.hstack((train_data, temp))

    n_series = 9
    target_row = 9

    mean_x = np.mean(train_data, axis = 1)
    std_x = np.std(train_data, axis = 1)

    for i in range(n_parameters):
        if std_x[i] != 0:
            train_data[i, :] = (train_data[i, :] - mean_x[i])/std_x[i]

    final_data = []


    for j in range(train_data.shape[1] - n_series):
        key = train_data[:, j:j+n_series].reshape(18*9, )
        value = train_data[target_row, j+n_series]
        final_data.append([key, value])

    return final_data, [mean_x, std_x]
