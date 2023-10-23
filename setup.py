#!/usr/bin/env python

from distutils.core import setup

setup(name='env',
      version='0.1-alpha',
      description='A library to simulate a trading environment',
      author='Federico Caprini',
      author_email='federico.caprini@protonmail.com',
      url='github.com/yrenum/env',
      packages=['env'],
      install_requires=["stockholm",
                        "gymnasium",
                        "pandas",
                        "numpy",
                        "dask",
                        "pandas-ta"],
      )
