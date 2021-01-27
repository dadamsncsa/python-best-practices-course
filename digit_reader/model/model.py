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
from helpers import num_classes, input_shape


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
