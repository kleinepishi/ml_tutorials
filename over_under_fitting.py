from __future__ import absolute_import, division, print_function, unicode_literals

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