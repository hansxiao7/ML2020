import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# Load dataset
mnist_ds_train, mnist_ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)
svhn_ds_train, svhn_ds_test = tfds.load('svhn_cropped', split=['train', 'test'], shuffle_files=True)

n_mnist_train = tf.data.experimental.cardinality(mnist_ds_train).numpy()
n_mnist_test = tf.data.experimental.cardinality(mnist_ds_test).numpy()

n_svhn_train = tf.data.experimental.cardinality(svhn_ds_train).numpy()
n_svhn_test = tf.data.experimental.cardinality(svhn_ds_test).numpy()

# To be fair for these 2 tasks, the training data are both 50000 cases, and the test data are 10000 cases
mnist_train_data = []
mnist_train_label = []
mnist_test_data = []
mnist_test_label = []

# The image size for MNIST is (28, 28, 1). The image size for SVHN is (32, 32, 3). This step is to transfer SVHN image
# to the same size of MNIST images.
for example in mnist_ds_train.take(50000):
    img, label = example["image"], example["label"]
    img = tf.cast(img, tf.float32) / 255.
    mnist_train_data.append(img)
    mnist_train_label.append(label)

for example in mnist_ds_test.take(10000):
    img, label = example["image"], example["label"]
    img = tf.cast(img, tf.float32) / 255.
    mnist_test_data.append(img)
    mnist_test_label.append(label)

svhn_train_data = []
svhn_train_label = []
svhn_test_data = []
svhn_test_label = []

for example in svhn_ds_train.take(50000):
    img, label = example["image"], example["label"]
    img = tf.keras.preprocessing.image.array_to_img(img)
    img = img.convert(mode='L')
    img = img.resize(size=(28, 28))
    img = tf.keras.preprocessing.image.img_to_array(img)

    img = tf.cast(img, tf.float32) / 255.
    svhn_train_data.append(img)
    svhn_train_label.append(label)

for example in svhn_ds_test.take(10000):
    img, label = example["image"], example["label"]
    img = tf.keras.preprocessing.image.array_to_img(img)
    img = img.convert(mode='L')
    img = img.resize(size=(28, 28))
    img = tf.keras.preprocessing.image.img_to_array(img)

    img = tf.cast(img, tf.float32) / 255.
    svhn_test_data.append(img)
    svhn_test_label.append(label)

np.save('mnist_train_data.npy', mnist_train_data)
np.save('mnist_train_label.npy', mnist_train_label)

np.save('mnist_test_data.npy', mnist_test_data)
np.save('mnist_test_label.npy', mnist_test_label)

np.save('svhn_train_data.npy', svhn_train_data)
np.save('svhn_train_label.npy', svhn_train_label)

np.save('svhn_test_data.npy', svhn_test_data)
np.save('svhn_test_label.npy', svhn_test_label)

print(tf.shape(mnist_train_data[0]))
print(tf.shape(svhn_train_data[0]))