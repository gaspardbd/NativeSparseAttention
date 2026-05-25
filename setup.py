# -*- coding: utf-8 -*-

import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


def get_package_version():
    with open(Path(os.path.dirname(os.path.abspath(__file__))) / 'native_sparse_attention' / '__init__.py') as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    return ast.literal_eval(version_match.group(1))


setup(
    name='native_sparse_attention',
    version=get_package_version(),
    description='Trtion kernels for Native Sparse Attention',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yentl Collin, Gaspard Beaudoin',
    author_email='yentl.collin@universite-paris-saclay.fr',
    url='https://github.com/YentlCollin/native-sparse-attention',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.5',
        'transformers>=4.45.0',
        'triton>=3.0',
        'datasets>=3.3.0',
        'einops',
        'ninja'
    ],
    extras_require={
        'conv1d': ['causal-conv1d>=1.4.0']
    }
)
