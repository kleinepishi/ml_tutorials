from __future__ import absolute_import, division, print_function, unicode_literals

# import Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libs
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.keras.__version__)

# import fashion_mnist db
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

len(train_labels)

train_labels

test_images.shape

len(test_labels)

# data must be preprocessed before training
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale values to a range of 0 to 1 before feeding to NNM, divide by 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# display first 25 images from training set, with class name below image
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# basic building block: layer
# extract representation from data fed to them
# most DL is made up of chaining simple layers together (i.e tf.keras.layers.Dense)
model = keras.Sequential([
    # first layer transforms format of image from 2d array to 1d array of 28*28=784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    # after flattening, dense (fully-connected) neural layers are used
    # first has 128 nodes (neurons)
    keras.layers.Dense(128, activation=tf.nn.relu),
    # second is 10 node softmax lyer, returns 10 prob scores that sum to 1
    # each node has a score, indicating prob that current img is a member of one of the 10 classes
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# now to compile the model
# define loss function - how accurate the model is during training (want to minimize)
# optimizer - how model is updated based on data it sees and loss function
# metrics - used to monitor training and testing step (in this case, uses 'accuracy')
model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

# start to train NNM
# model learns to associate images and labels
# make predictions about a test set, then verify
model.fit(train_images, train_labels, epochs=5)

# evaluate accuracy of model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# now that we have trained the model, we can make predictions
predictions = model.predict(test_images)

# first prediction
predictions[0]
print('Predictions:', predictions[0])

# prediction is an array of 10 numbers, describe confidence of model
# that image corresponds to each of 10 different articles of clothing
# can observe which label has highest confidence value
np.argmax(predictions[0])
print('Highest confidence:', np.argmax(predictions[0]))

# check test label for correctness
test_labels[0]


