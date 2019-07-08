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