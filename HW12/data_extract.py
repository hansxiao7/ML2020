import tensorflow as tf
import numpy as np
import os
from PIL import Image

train_dir = 'ml2020spring-hw12/real_or_drawing/train_data/'

train_data = []
train_label = []
for i in range(10):
    x = []
    curr_train_dir = train_dir + str(i) + '/'

    for file in os.listdir(curr_train_dir):
        img = Image.open(curr_train_dir + file)
        img = img.resize((28, 28))
        img = img.convert(mode='L')
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.

        x.append(img_array)

    y = np.ones(len(x)) * i
    train_data.extend(x)
    train_label.extend(y)

np.save('train_data.npy', train_data)
np.save('train_label.npy', train_label)
print(tf.shape(train_data))
print(tf.shape(train_label))

test_dir = 'ml2020spring-hw12/real_or_drawing/test_data/0/'

test_data = []
for file in os.listdir(test_dir):
    img = Image.open(test_dir + file)
    img = img.resize((28, 28))
    img = img.convert(mode='L')
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.

    test_data.append(img_array)

np.save('test_data.npy', test_data)
print(tf.shape(test_data))
