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
from digit_reader.model.helpers import num_classes, input_shape


class MNISTModel:
    def __init__(self):
        """Create initial model and print summary"""
        self.model = keras.Sequential(
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

        self.model.summary()

    def train_model(self, x_train, y_train, epochs=5):
        """Train the model with specified training data

        Args:
            x_train: MNIST images training set
            y_train: MNIST image labels training set

        """
        batch_size = 128

        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
        )

    def evaluate_model(self, x_test, y_test):
        """Evaluate the model and return the score

        Returns:
            A tuple containing the loss and accuracy

        """
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        return score
    
    def classify_image(self, image):
        """Make predictions on what the label is for a given image

        Returns:
            A label integer which the image most likely belongs to

        """
        
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(np.array([image]))
        return np.argmax(predictions[0])
