from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# download imbd dataset
# multi-hot encoding the lists means turning them into vectors of 0s and 1s
# example, turning sequence [3, 5] into a 10000 dimensional vector which would be all 0s
# aside for indices 3 and 5, which would be ones
# used for quickly demostrating how overfitting happens, and ways to combat it
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# now look at a resultant multi-hot vector
plt.plot(train_data[0])

# easiest way to prevent overfitting is to decrease model size (# of learnable parameters)
# model size is determined by number of layers and numbers of units per layer
# learnable parameters in model are known as "capacity"
# a model with more parameters will have greater "memorization capacity"
# will therefore be able to more easily learn perfect dictionary-like mapping between training samples and targets
# issue with this is that the mapping has no generalization abilities
# useless when attempting to make predictions on unseen data
# need to find sweet spot of generalization, not fitting to training data

# another note, if network has limited memorization resources, will be unable to learn mapping easily
# to minimize loss, will need to learn compressed representations that have higher predictive power
# if you make model too small, will have trouble fitting to training data
# must balance between too much and not enough capacity

# to start, best to begin with few layers and parameters and then build up from there
# until you begin to see diminishing returns on validation loss
# for example, first start with simple model using just Dense layers as baseline
# will create smaller and larger version, then compare
baseline_model = keras.Sequential([
    # 'input_shape' is only required so '.summary' will function
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
baseline_model.summary()

baseline_history = baseline_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# now generate a similar model below with less hidden units
smaller_model = keras.Sequential([
    # 'input_shape' is only required so '.summary' will function
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
smaller_model.summary()

smaller_history = smaller_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# for comparison purposes, we make an even larger model than baseline to demonstrate how quickly overfitting occurs
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()

bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

# plot training and validation loss
# solid lines denote training loss
# dashed lines show validation loss
# lower validation loss indicates better model

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])