#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

req = os.path.dirname(os.path.realpath(__file__))
req = os.path.join(req, "requirements.txt")
requirements = []
if os.path.isfile(req):
    with open(req) as f:
        requirements = f.read().splitlines()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Nikhil Gupta",
    author_email='me@nikhgupta.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Portfolio backtesting, paperr and live trading tool.",
    entry_points={
        'console_scripts': [
            'backfolio=backfolio.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='backfolio',
    name='backfolio',
    packages=find_packages(include=['backfolio', 'backfolio.core', 'backfolio.strategy']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nikhgupta/backfolio',
    version='1.5.8',
    zip_safe=False,
)
