import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Data images are 224x224 RGB
# Load VGG16 as the proxy network for attack
VGG16 = tf.keras.applications.VGG16()
# VGG16.summary()

# Load data images
image_path = 'data/images/'
image_data = []
for file_name in os.listdir(image_path):
    img_raw = Image.open(image_path+file_name)
    img_array = np.array(img_raw)
    img_array = np.reshape(img_array, (224, 224, 3))
    image_data.append(img_array)

image_data = tf.convert_to_tensor(image_data)

# Load original data labels
label_file_path = 'data/labels.csv'
true_labels = np.loadtxt(label_file_path, delimiter=',', skiprows=1, usecols=3)

# Calculate the original losses before attacking
estimated_label = VGG16.predict(image_data)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
original_loss = loss(true_labels, estimated_label).numpy()

# Use FGSM to generate attacked images
epislon = 15
image_number = 200
attacked_image = []
for i in range(2):#image_number):
    x = image_data[i]
    x = np.reshape(x, (1, 224, 224, 3))
    x = tf.Variable(x, dtype=tf.float32)
    y = true_labels[i]
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss_func = loss(y, VGG16(x))
    grads = tape.gradient(loss_func, x)
    sign_grads = np.sign(grads)
    sign_grads = np.reshape(sign_grads, (224, 224, 3))
    x_attacked = image_data[i] + epislon * sign_grads
    attacked_image.append(x_attacked)

# for j in range(len(attacked_image)):
for j in range(2):
    attacked_x = tf.keras.preprocessing.image.array_to_img(attacked_image[j])
    tf.keras.preprocessing.image.save_img('data/attacked/'+str(j)+'.png', attacked_x)

# attacked_image = tf.convert_to_tensor(attacked_image)
# attacked_label = VGG16.predict(attacked_image)
# attacked_loss = loss(true_labels, attacked_label).numpy()
# print('Original loss is: ' + str(original_loss))
# print('Attacked loss is: ' + str(attacked_loss))

preds = tf.keras.applications.vgg16.decode_predictions(VGG16.predict(np.reshape(image_data[0], (1, 224, 224, 3))), top=5)
print(preds)
preds = tf.keras.applications.vgg16.decode_predictions(VGG16.predict(np.reshape(attacked_image[0], (1, 224, 224, 3))) , top=5)
print(preds)