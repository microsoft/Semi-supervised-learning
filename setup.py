# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path
import pathlib

import pkg_resources

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='semilearn',
    version='0.3.0',
    description='Unfied Semi-Supervised Learning Benchmark',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/Semi-supervised-learning',
    author='Yidong Wang*, Hao Chen*, Yue Fan*, Wang Sun, Ran Tao, Wenxin Hou, Renjie Wang, Heli Qi, Zhen Wu,' 
           'Satoshi Nakamura, Wei Ye, Marios Savvides, Bhiksha Raj, Takahiro Shinozaki, Bernt Schiele, Jindong Wang, Xing Xie, Yue Zhang',
    author_email='yidongwang37@gmail.com, haoc3@andrew.cmu.edu, yuefan@mpi-inf.mpg.de, jindwang@microsoft.com',

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch semi-supervised-learning',
    packages=find_packages(exclude=['preprocess', 'saved_models', 'data', 'config']),
    include_package_data=True,
    # install_requires=['torch >= 1.8', 'torchvision', 'torchaudio', 'transformers', 'timm', 'progress', 'ruamel.yaml', 'scikit-image', 'scikit-learn', 'tensorflow', ''],
    install_requires=install_requires,
    python_requires='>=3.8',
)
