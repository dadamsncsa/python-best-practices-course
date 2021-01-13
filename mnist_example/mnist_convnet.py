"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10
"""Defines how many possible numbers to classify"""

input_shape = (28, 28, 1)
"""Defines the shape of the input images, 28x28 with one color channel"""


def prepare_data():
    """Load dataset and preprocess data

    Returns:
        A tuple of pre-processed training data

    """

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Create a simple convolutional Sequential model with 7 layers

    Returns:
        The built model

    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    return model


def train_model(model, x_train, y_train):
    """Train a model with specified training data

    Args:
        model: A keras Model
        x_train: MNIST images training set
        y_train: MNIST image labels training set

    Returns:
        A trained model

    """
    batch_size = 128
    epochs = 5

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )
    return model


def evaluate_model(model, x_test, y_test):
    """Evaluate a given model and return the score

    Returns:
        A tuple containing the loss and accuracy

    """
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score
