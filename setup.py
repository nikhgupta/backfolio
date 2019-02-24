#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

__version__ = '1.7.5'

requirements = [
        'Click>=6.0',
        'beautifulsoup4==4.6.3',
        'ccxt==1.17.485',
        'empyrical==0.5.0',
        'matplotlib==3.0.1',
        'pandas==0.23.4',
        'pandas-datareader==0.7.0',
        'requests==2.20.1',
        'scikit-learn==0.20.0',
        'scipy==1.1.0',
        'TA-Lib==0.4.17',
        'tabulate==0.8.2',
        'numpy==1.16',
        'pyprind']

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
    packages=find_packages(include=['backfolio', 'backfolio.core',
        'backfolio.strategy', 'backfolio.strategy.mixins']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nikhgupta/backfolio',
    version=__version__,
    zip_safe=False,
)
