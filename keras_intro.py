from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# require the above, otherwise will break

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

# build simple sequential model
# assemble layers to build models
# model is usually a graph of layers, common type of model is stack of layers

model = tf.keras.Sequential()
# adds a densely-connected layer with 64 units to model:
model.add(layers.Dense(64, activation='relu'))
# add second layer
model.add(layers.Dense(64, activation='relu'))
# add softmax layer w/ 10 output units:
model.add(layers.Dense(10, activation='softmax'))

# configure layers
# activation: set activation function for layer, param specified by name of built-in function,
# by default, not activation is applied
# kernel_intializer and bias_initializer: init schemes that create layers weights (kernel and bias)
# param is callable object, default is "Glorot uniform" initializer
# kernel_regularizer and bias_regularizer: regularization schemes that apply the layers weights
# L1 or L2 regularization, default is none

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# train and evaluate
# configure learning process of model by calling compile method

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# optimizer: specifies training procedure, 
# i.e. instances like tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.GradientDescentOptimizer
# loss: function that you want to minimize during optimization.  
# Common choices: mean square error (mse), categorical_crossentropy, binary_crossentropy
# passed as callabel object by tf.keras.losses
# metrics: used to monitor training, usually string names or callables from tf.keras.metrics

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# when working with small datasets, use in-memory NumPy arrays to train and evaluate models
# model is fit to trainting data using "fit" method

import numpy as np

def random_one_hot_labels(shape):
  n, n_class = shape
  classes = np.random.randint(0, n_class, n)
  labels = np.zeros((n, n_class))
  labels[np.arange(n), classes] = 1
  return labels

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

# tf.keras.Model.fit takes 3 important arguments
# epochs: training is structured into epochs, epoch is one iteration over entire input data (done in smaller batches)
# batch_size: when passed NumPy data, models slices it into smaller batches, iterates over during training
# above integer specifiees size of each batch
# validation_data: data used to monitor performance.  Passing the arg ( tuple of inputs and labels)
# allows model to display loss and metrics in inference mode for passed data, at end of each epoch

# validation data example:

import numpy as np

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

# input tf.data datasets

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)

# the above fit method uses steps_per_epoch argument (# of training steps model runs before moving to next epoch)
# since Dataset gives us batches of data, snippet does not need batch_size

# can also use Datasets for validation
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

# evaluate and predict

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)

# predict output of last layer in inference for data provided as numpy array
result = model.predict(data, batch_size=32)
print(result.shape)