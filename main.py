"""Main function for running mnist example"""
from digit_reader.model.model import MNISTModel
from digit_reader.model.helpers import prepare_data

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = prepare_data()
    model = MNISTModel()
    model.train_model(x_train, y_train, epochs=2)
    model.evaluate_model(x_test, y_test)
