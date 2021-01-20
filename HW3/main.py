import matplotlib.pyplot as plt
import numpy as np
import os

import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 128
img_width = 128

data_dir = 'Images/training'

# generate the label list
# label_list = []
# for root, dirs, files in os.walk(data_dir):
#     for names in files:
#         curr_label = names.split('_')
#         label_list.append(int(curr_label[0]))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='Images/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.3,
    subset='training',
    seed=1
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='Images/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.3,
    subset='validation',
    seed=1
)

# build CNN
from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

num_classes = 11

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255), #128,128,3
  layers.Conv2D(64, 3, activation='relu', padding='same'), #126,126,64
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #62,62,64

  layers.Conv2D(128, 3, activation='relu', padding='same'), #60, 60, 128
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #30,30,128

  layers.Conv2D(256, 3, activation='relu', padding='same'), #30,30,256
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),#14,14,256

  layers.Conv2D(512, 3, activation='relu', padding='same'),#12,12,512
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),#6,6,512

  layers.Conv2D(512, 3, activation='relu', padding='same'),#4,4,512
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),#2,2,512

  layers.Flatten(),
  layers.Dense(1024, activation='relu'),
  layers.Dense(512, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=30
)

model.save('CNN.h5')