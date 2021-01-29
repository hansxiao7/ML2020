from PIL import Image
import numpy as np
import tensorflow as tf

image_0 = '0_0.jpg'
image_1 = '1_29.jpg'
image_2 = '2_9.jpg'
image_3 = '3_25.jpg'
image_5 = '5_27.jpg'

# Load and resize image for CNN use
img_height = 128
img_width = 128

curr_image = Image.open(image_5)
class_number = 5


curr_image = curr_image.resize((img_width, img_height))
input_data = np.array(curr_image)
input_data = np.reshape(input_data, (1, img_width, img_height, 3))

# Load CNN model and check the results
cnn_model = tf.keras.models.load_model('CNN.h5')
cnn_model.summary()

image_class = cnn_model.predict(input_data)
predicted_class = np.argmax(image_class)

print('The class of the current image is: ' + str(predicted_class))

# Initialize a saliency map
input_data = tf.constant(input_data, dtype=tf.float64)
with tf.GradientTape() as tape:
    tape.watch(input_data)
    loss = cnn_model(input_data, training = False)[0, class_number]
grads = tape.gradient(loss, input_data)

grads = (grads - np.min(grads))/(np.max(grads) - np.min(grads))
grads = grads * 255
grads = np.array(grads, dtype=int)
grads = np.reshape(grads, (128, 128, 3))

saliency_map = tf.keras.preprocessing.image.array_to_img(grads)
saliency_map = saliency_map.resize((512, 512))
tf.keras.preprocessing.image.save_img('saliency_map_5.jpeg', saliency_map)
