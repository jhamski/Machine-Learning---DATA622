#!/usr/bin/env python

# Code citation  and tutorial used as a starting point:
# http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#Set a random seed in order to have consistent results
seed = 9103
numpy.random.seed(seed)

def load_reshape():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return X_train, X_test, y_train, y_test, num_classes

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(
        32, 5, 5,
        border_mode='valid',
        input_shape=(1, 28,28)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy'])

    return model

# load and reshape the MNIST dataset
X_train, X_test, y_train, y_test, num_classes = load_reshape()

print "Running the tutorial model."
"""
# build the model
model = cnn_model()

# Fit the model
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    nb_epoch=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
"""
#CNN Error: 1.42%

"""
Running in the terminal this script outputs CNN Error: 1.42%.
Looking back at my feed forward neural network script:
https://github.com/jhamski/Machine-Learning---DATA622/blob/master/WK_8/JHamski_week_8_MNIST.ipynb
I found an error of 1.58%. Therefore, the CNN here represents a
slight improvement in accuracy even though it used half as many epochs as
my FNN model (10 vs 20). Below, I attempt to improve upon the
tutorial's CNN model.
"""

def cnn_model_jh():
    model = Sequential()
    model.add(Convolution2D(
        32, 10, 10,
        border_mode='valid',
        input_shape=(1, 28,28)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics=['accuracy'])

    return model

print "Running the modified model."

# build the model
model_jh = cnn_model_jh()

# Fit the model
model_jh.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    nb_epoch=16, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model_jh.evaluate(X_test, y_test, verbose=0)
print("Improved CNN Error: %.2f%%" % (100-scores[1]*100))

# Improved CNN Error: 1.26%

"""
The improvements I made and my reasoning is as follows:

1) Increased convolution kernal size
The variability MNIST numeric character images is fairly "sparse", meaning the
information in the structure is dense. There just isn't much detail needed to
distinguish the structural differences between numbers. My hypothesis is that
increasing the size of the first convolutional filter will better focus on the
signal inherit in the structure of a number character, not the noise inherit in
handwriting.

Note that an experiment where the convolutional kernal was increased to 20 x 20
pixes increased the error rate by about 0.40%, so it appears this parameters
has an optimal value somewhere between 5 x 5 and 20 x 20.

2) Increasing number of epochs
Increasing the number of epochs to 16 lead to a marginal increase in model
accuracy.

3) Adding an additional hidden layer to the neural network
Adding an additional hidden layer with: model.add(Dense(50)) increased the model
accuracy. This is an important point - convolutional filters are like a preprocesing
step, there is still a neural network model to tune after the convolutions.

4) Tried but not implemented:Adding a second convolutional filter
Adding a second convolutional filter was tried with several configurations.
However, it resulted in poorer accuracy, possible because the MNIST dataset
images are fairly small. This method may work better for larger images.

My model improved accuracy by 0.42%.
"""
