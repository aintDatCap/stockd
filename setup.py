#!/usr/bin/env python

from distutils.core import setup

setup(name='stockazzo',
      version='0.1-alpha',
      description='A library to simulate a trading environment',
      author='Federico Caprini',
      author_email='federico.caprini@protonmail.com',
      url='github.com/yrenum/stockazzo',
      packages=['stockazzo'],
      install_requires=["stockholm",
                        "gymnasium",
                        "pandas",
                        "numpy",
                        "dask",
                        "pandas-ta"],
      )
