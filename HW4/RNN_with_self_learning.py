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
  epochs=1
)

# Self learning with unlabelled data
threshold = 0.8
for i in range(10):
    new_train_data = []
    new_train_label = []
    tf.random.shuffle(nolabel_data)
    for j in range(1000):
        x = nolabel_data[j]
        y = model.predict(x)[0] # Since bidirectional layer is used, the class prediction is the first value
        if y >= threshold:
            new_train_data.append(x)
            new_train_label.append(int(1))
        elif y <= (1-threshold):
            new_train_data.append(x)
            new_train_label.append(0)
    new_train_label = tf.cast(new_train_label, dtype=tf.int64)
    print('Prediction done for 1000 cases!')

    new_train_dataset = tf.data.Dataset.from_tensor_slices((new_train_data, new_train_label))
    batch_size = 100
    new_train_dataset = new_train_dataset.batch(batch_size)

    combined_dataset = train_dataset.concatenate(new_train_dataset)

    model.fit(
        combined_dataset,
        validation_data=val_dataset,
        epochs=1
    )

