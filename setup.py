#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()

install_requires = [
    'numpy>=1.18.5',
    'scikit_image>=0.16.2',
    'torch>=1.6',
    'torchvision>=0.6.0'
]

setuptools.setup(
    name='deepMRI',
    version='0.0.1',
    description='A deep learning toolbox for parallel MRI reconstruction',
    author='Yi Zhang',
    author_email='yizhang.dev@gmail.com',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/yzhang-dev/deepMRI',
    download_url='https://github.com/yzhang-dev/deepMRI',
    packages=[
        'deepmri'
    ],
    keywords=[
        'MRI',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    license='MIT',
    python_requires='>=3.7',
    install_requires=install_requires,
)
