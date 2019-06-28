from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

# download imdb dataset locally
imdb = keras.datasets.imdb
# num_words=10000 keeps the 10000 most frequent words in training data
# required to discard rare words to preseve manageability of data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# explore data
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1])

# convert int back to words

# dictionary mapping word to int index
word_index = imdb.get_word_index()

# initial indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# use decode_review to display text for first review
decode_review(train_data[0])

# prep the data for training
# pad arrays so they all have same length, generate integer tensor shape comprised of max_length + num_reviews
# use embedding layer to handle shape as first layer in network
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

# examine length of examples
len(train_data[0]), len(train_data[1])

# look at first (padded) review
print(train_data[0])

# input data is composed of an array of word-indicies, label predicts either 0 or 1
# below is model 
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# for training, a model requires a loss function as well as an optimizer
# since this is binary classification problem, use binary_crossentropy loss function (other options available)
# measures "distance" between probability distributions
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# when training, important to verify accuracy of model on unseen data
# generate validation set of some # of examples from original testing data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train model
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# evaluate model
results = model.evaluate(test_data, test_labels)
print(results)

# generate graph of accuracy and loss of time
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" = "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# "b" = "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf() # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
