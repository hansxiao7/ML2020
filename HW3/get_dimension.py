
import tensorflow as tf
from tensorflow.keras import layers

input_shape = (4, 128, 128, 3)
x = tf.random.normal(input_shape)
y = tf.keras.Sequential([
  layers.Conv2D(64, 3, activation='relu', padding='same'), #128,128,64
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #62,62,64

  layers.Conv2D(128, 3, activation='relu', padding='same'), #60, 60, 128
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #30,30,128

  layers.Conv2D(256, 3, activation='relu', padding='same'), #30,30,256
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),#14,14,256

  layers.Conv2D(512, 3, activation='relu', padding='same'),#12,12,512
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),#6,6,512

  layers.Conv2D(512, 3, activation='relu', padding='same'),#4,4,512
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),#2,2,512
])(x)

print(y.shape)
