import numpy as np
import tensorflow as tf

teacher_model = tf.keras.models.load_model('ml2020spring-hw7/food-11/CNN.h5')
teacher_model.summary()

# Get the teacher model outputs
batch_size = 1
img_height = 128
img_width = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='ml2020spring-hw7/food-11/training/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='ml2020spring-hw7/food-11/validation/',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

train_size = 9866

teacher_output = teacher_model.predict(train_ds)
teacher_output = tf.reshape(teacher_output, (tf.shape(teacher_output)[0], 1, tf.shape(teacher_output)[1]))


# Build training data for student
input_data = []
for data, labels in train_ds.take(train_size):
    input_data.append(data)

student_train_ds = tf.data.Dataset.from_tensor_slices((input_data, teacher_output))

# Build student model
from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

num_classes = 11

student_model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255), #128,128,3
  layers.Conv2D(64, 3, activation='relu', padding='same'), #126,126,64
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #62,62,64

  layers.Conv2D(128, 3, activation='relu', padding='same'), #60, 60, 128
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #30,30,128

  layers.Flatten(),
  layers.Dense(1024, activation='relu'),
  layers.Dense(512, activation='relu'),
  layers.Dense(num_classes)
])

student_model.compile(
  optimizer='adam',
  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

student_model.fit(
  student_train_ds,
  epochs=10
)