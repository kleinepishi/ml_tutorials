# Model progress can be saved during and after training
# allows for resumption of model and avoidance of long training times
# also allows for ssaving and distribution of model(s)

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# build a simple model
# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

  return model

# Create a basic model instance
model = create_model()
model.summary()

# save checkpoints during training
# main use case is to save cehckpoints during and at the end of training
# allows you to use a trained model without having to retrain it
# also lets you pick up where you left off in case process was interrupted
# tf.keras.callbacks.ModelCheckpoint is a callback that performs this task

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# generate checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
model = create_model()

model.fit(train_images, train_labels, epochs=10,
            validation_data= (test_images,test_labels),
            callbacks = [cp_callback])
# may throw warnings related to saving state of optimizer

# make a new and untrained model
# when restoring a model fromjust weigths, model must have same architecture as original
# since its identical, can share weights despite being a differente instance of model

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# load weights from checkpoint and re-evaluate

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))