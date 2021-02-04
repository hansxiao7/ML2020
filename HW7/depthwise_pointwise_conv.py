import tensorflow as tf
from tensorflow.keras import layers


input_flow = tf.keras.layers.Input(shape=(128, 128, 3))
one = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', input_shape=(None, 128, 128, 1))(
    tf.expand_dims(input_flow[:, :, :, 0], axis=-1))
two = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', input_shape=(None, 128, 128, 1))(
    tf.expand_dims(input_flow[:, :, :, 1], axis=-1))
three = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', input_shape=(None, 128, 128, 1))(
    tf.expand_dims(input_flow[:, :, :, 2], axis=-1))
trail_layer = tf.keras.layers.concatenate([one, two, three], axis=-1)


input_shape = (4, 128, 128, 3)
x = tf.random.normal(input_shape)
model = tf.keras.Model(inputs=input_flow, outputs=trail_layer)
model.summary()

model2 = tf.keras.Sequential([
   model,
   layers.Conv2D(64, 1, activation='relu', padding='same'), #126,126,64
   layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
 ])
model2.summary()

