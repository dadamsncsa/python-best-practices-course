import os
import pytest

class TestForFiles():
    def test_for_license(self):
        assert os.path.isfile("LICENSE")

    def test_for_gitignore(self):
        assert os.path.isfile(".gitignore")
    
    def test_for_readme(self):
        assert os.path.isfile("README.md")