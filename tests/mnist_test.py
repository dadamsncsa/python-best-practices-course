from mnist_example.mnist_convnet import prepare_data, build_model, train_model, evaluate_model

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
    model = build_model()
    # test if model weights were generated
    assert model.get_weights() is not None
    # test if each of the layer connections has weights
    # there are 7 layers, so 6 weights
    assert len(model.get_weights()) == 6


def test_model_score():
    (x_train, y_train), (x_test, y_test) = prepare_data()
    model = build_model()
    trained_model = train_model(model, x_train, y_train)
    score = evaluate_model(trained_model, x_test, y_test)
    # test if accuracy was greater than 90%
    assert score[1] >= .9

