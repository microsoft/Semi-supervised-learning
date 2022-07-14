# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='usb',
    version='0.0.0',
    description='Unfied Semi-Supervised Learning Benchmark',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/Semi-supervised-learning',
    # author='',
    # author_email='',

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch semi-supervised-learning',
    packages=find_packages(exclude=['preprocess', 'saved_models', 'data', 'config']),
    include_package_data=True,
    install_requires=['torch >= 1.8', 'torchvision', 'torchaudio', 'transformers'， 'timm'],
    python_requires='>=3.7',
)