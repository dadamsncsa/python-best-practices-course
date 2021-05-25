import os
import pytest


class TestSetup():
    def test_fields(self):
#   FIELDS TO CHECK FOR
#       python_requires='>=3.0',
#     install_requires=[
#         'django',
#         'pandas'
#     ]
#     name="mnist_example-nanoMFG", # Replace with your own username
#     version="0.0.1",
#     author="Example Author",
#     author_email="author@example.com",
#     description="A small example package",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/pypa/sampleproject",
#     packages=setuptools.find_packages()
        with open('setup.py') as setup_file:
            all_satisfied = True
            all_fields = {
              'python_requires' : False,
              'install_requires' : False,
              'name' : False,
              'version' : False,
              'author' : False,
              'author_email' : False,
              'description' : False,
              'long_description' : False,
              'long_description_content_type' : False,
              'url' : False,
              'packages' : False}
            for line in setup_file:
                for key in all_fields:
                    if key in line:
                        all_fields[key] = True
            for key in all_fields:
                if not all_fields[key]:
                    all_satisfied = False
        assert all_satisfied
            
