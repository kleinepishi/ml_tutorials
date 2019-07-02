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
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']

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
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# basic building block: layer
# extract representation from data fed to them
# most DL is made up of chaining simple layers together (i.e
# tf.keras.layers.Dense)
model = keras.Sequential([
    # first layer transforms format of image from 2d array to 1d array of
    # 28*28=784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    # after flattening, dense (fully-connected) neural layers are used
    # first has 128 nodes (neurons)
    keras.layers.Dense(128, activation=tf.nn.relu),
    # second is 10 node softmax layer, returns 10 prob scores that sum to 1
    # each node has a score, indicating prob that current img is a member of
    # one of the 10 classes
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# now to compile the model
# define loss function - how accurate the model is during training (want to minimize)
# optimizer - how model is updated based on data it sees and loss function
# metrics - used to monitor training and testing step (in this case, uses
# 'accuracy')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

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

# graph to observe entire set of 10 class predictions


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label]),
        color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# now examine 0th image, predictions, and prediction array
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# next, plot several images with predictions
# correct are blue, incorrect red
# number gives % (out of 100) for the predicted label
# can be wrong even with high confidence

# plot first x test images, predicted label, and true label
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# pull image from test dataset
img = test_labels[0]
print(img.shape)

# add image to a batch where it is only member
img = (np.expand_dims(img, 0))
print(img.shape)

# predict following image:
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

# grab predictions for only image in batch
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
