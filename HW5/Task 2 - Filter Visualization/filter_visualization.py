import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained CNN model
cnn_model = tf.keras.models.load_model('CNN.h5')
cnn_model.summary()

# Get layer names and layers
layer_dict = dict([(layer.name, layer) for layer in cnn_model.layers])
print(layer_dict.keys())

visualized_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']

for curr_layer_name in visualized_layers:
    curr_layer = layer_dict[curr_layer_name]
    feature_extractor = tf.keras.Model(inputs=cnn_model.inputs, outputs=curr_layer.output)


    # define loss_function
    def get_loss(image, filter_index):
        outputs = feature_extractor(image)
        loss = tf.reduce_mean(outputs[:, :, :, filter_index])
        return loss


    # Initialization
    learning_rate = 10
    iteration = 100

    for filter_index in range(5):
        new_image = np.random.random(size=(1, 128, 128, 3)) * 20 + 128
        new_image = tf.constant(new_image, dtype=tf.float64)

        for i in range(iteration):
            with tf.GradientTape() as tape:
                tape.watch(new_image)
                loss = get_loss(new_image, filter_index)
            grads = tape.gradient(loss, new_image)
            grads = tf.math.l2_normalize(grads)
            grads = tf.where(grads < 0, 0, grads)
            new_image += learning_rate * grads

        # Image post-processing
        new_image = np.reshape(new_image, (128, 128, 3))
        new_image = (new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image))

        # convert to RGB array
        new_image *= 255
        x = np.clip(new_image, 0, 255).astype('uint8')

        filter_map = tf.keras.preprocessing.image.array_to_img(new_image)
        filter_map = filter_map.resize((512, 512))
        tf.keras.preprocessing.image.save_img(curr_layer_name + '_' + str(filter_index) + '.jpeg', filter_map)
