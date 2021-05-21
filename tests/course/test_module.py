import os
import pytest

class TestModules():
    def test_initpy(self):
        assert os.path.isfile('src/digit_reader/__init__.py')
        assert os.path.isfile('src/digit_reader/model/__init__.py')

    def test_skeleton_code(self):
        with open('src/digit_reader/model/model.py') as model_file:
            model_exists = False
            train_exists = False
            evaluate_exists = False
            for line in model_file:
                if 'class MNISTModel:' in line:
                    model_exists = True
                elif 'def train_model(self' in line:
                    train_exists = True
                elif 'def evaluate_model(self' in line:
                    evaluate_exists = True
            assert model_exists and train_exists and evaluate_exists        

    def test_docstrings(self):
        with open('src/digit_reader/model/model.py') as model_file:
            docstrings = 0
            for line in model_file:
                if '\"\"\"' in line:
                    docstrings += 1
        assert docstrings >= 3
