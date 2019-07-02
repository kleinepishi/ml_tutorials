from __future__ import absolute_import, division, print_function, unicode_literals

# require the below, otherwise will break
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# require the above, otherwise will break

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# download needed data
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

# import using pandas
column_names = ['MPG',
                'Cylinders',
                'Displacement',
                'Horsepower',
                'Weight',
                'Acceleration',
                'Model Year',
                'Origin']
raw_dataset = pd.read_csv(dataset_path,
                          names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# clean the data
dataset.isna().sum()

# drop some extraneous rows
dataset = dataset.dropna()

# 'Origin' column is categorical, not numeric, need to convert to one-hot
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()

# split data into train and testing sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# inspect data
sns.pairplot(train_dataset[["MPG",
                            "Cylinders",
                            "Displacement",
                            "Weight"]],
             diag_kind="kde")

# examine overal statistics
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

# separate features from labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# normalize the data
# important to normalize features that use different scales and ranges
# without normalization, training is more difficult
# causes dependancy on choice of units used as input


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# build the model


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()

# examine model with .summary method
model.summary()

# test model
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print('Example result: ', example_result)

# display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# visualize model's training process using stats kept in history 
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

# resultant graph shows little improvement, degradation even in validation error after ~100 epochs
# update the model.fit to automatically stop training when validation scores stop improving
# use EarlyStopping callback, tests training condition for every epoch.

model = build_model()

# patientience parameter is threshold of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history)

# graph demonstrateds that on the validation set, average error is +/- 2 MPG
# check how well model generalizes through using test set, which was not used when training the model
# gives a baseline as what to expect the model to predict when using in the real world

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# lastly, predict MPG values using data from testing set
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# examine error distribution

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

# tutorial introduced:
# Mean Squared Error (MSE), common loss function used for regressions problems
# evaluation metrics used for regression differ from classification
# When numeric input data features have values with diff ranges, need to use feature normalization
# if not much training data, use small network with a couple hidden layers to prevent overfitting
# early stopping is useful to avoid overfitting