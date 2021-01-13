"""Main function for running mnist example"""
from mnist_example.mnist_convnet import (
    prepare_data,
    build_model,
    train_model,
    evaluate_model,
)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = prepare_data()
    model = build_model()
    trained_model = train_model(model, x_train, y_train)
    evaluate_model(trained_model, x_test, y_test)
