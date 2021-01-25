import numpy as np
import tensorflow as tf


# This file to read data and labels from original .txt files, and use 1-of-N hot encoder to transfer string to
# integer arrays.
def one_hot_encoder(train_file_labelled, train_file_nolabel, test_file):
    train_file = []

    # read labels, text and unlabelled text from the training file with labels
    with open(train_file_labelled, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' +++$+++ ')
            label = int(line[0])
            text = line[1]
            train_file.append([text, label])
    train_file = np.array(train_file)
    # Generate validation data set and train data set
    train_ratio = 0.7
    np.random.shuffle(train_file)
    n_train = int(np.shape(train_file)[0] * train_ratio)

    train_text = train_file[:n_train, 0]
    train_label = np.array(train_file[:n_train, 1]).astype(int)

    val_text = train_file[n_train: np.shape(train_file)[0], 0]
    val_label = np.array(train_file[n_train: np.shape(train_file)[0], 1]).astype(int)


    # Get text from unlabelled data
    text_no_label = []
    with open(train_file_nolabel, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            text_no_label.append(line)

    # Get text from test file
    test_text = []
    with open(test_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            test_text.append(line)

    # Preprocessing text to 1-and-N hot encode
    max_vocabulary = 100000
    max_sentence_length = 50

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_vocabulary,
        output_mode='int',
        output_sequence_length=max_sentence_length
    )

    total_text = np.concatenate((train_text, text_no_label, val_text, test_text))
    vectorize_layer.adapt(total_text)

    train_data = vectorize_layer(train_text)
    nolabel_data = vectorize_layer(text_no_label)
    val_data = vectorize_layer(val_text)
    test_data = vectorize_layer(test_text)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))

    batch_size = 100
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset, nolabel_data, test_data

# train_file_labelled = 'training_label.txt'
# train_file_nolabel = 'training_nolabel.txt'
# test_file = 'testing_data.txt'
# one_hot_encoder(train_file_labelled, train_file_nolabel, test_file)
