import numpy as np
import tensorflow as tf
from one_hot_encoder import one_hot_encoder

train_file_labelled = 'training_label.txt'
train_file_nolabel = 'training_nolabel.txt'
test_file = 'testing_data.txt'
train_dataset, val_dataset, nolabel_data, test_data = one_hot_encoder(train_file_labelled, train_file_nolabel,
                                                                      test_file)

max_vocabulary = 100000
embedding_output_length = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=max_vocabulary,
        output_dim=embedding_output_length,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=10
)
