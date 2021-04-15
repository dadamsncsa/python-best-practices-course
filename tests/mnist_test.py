from digit_reader.model.helpers import prepare_data
from digit_reader.model.model import MNISTModel

def test_failing():
    # this test should initially be failing for some reason, then the user will fix it
    # could be as simple as having it initially be 'assert False' or something more complex
    assert True


def test_x_shape():
    (x_train, y_train), (x_test, y_test) = prepare_data()
    # test if all training images were loaded
    assert x_train.shape[0] == 60000
    # test if all test images were loaded
    assert x_test.shape[0] == 10000
    # test if images are square
    assert x_train.shape[1] == x_train.shape[2]


def test_y_shape():
    (x_train, y_train), (x_test, y_test) = prepare_data()
    # test if all training labels were loaded
    assert y_train.shape[0] == 60000
    # test if all test labels were loaded
    assert y_test.shape[0] == 10000


def test_model_built():
    model = MNISTModel()
    # test if model was generated
    assert model is not None


def test_model_score():
    (x_train, y_train), (x_test, y_test) = prepare_data()
    model = MNISTModel()
    model.train_model(x_train, y_train, epochs=2)
    score = evaluate_model(x_test, y_test)
    # test if accuracy was greater than 90%
    assert score[1] >= .9
    
def test_classify_image():
    (x_train, y_train), (x_test, y_test) = prepare_data()
    model = MNISTModel()
    assert model.classify_image(x_test[0]) == 9
        
