#!/usr/bin/env python

from distutils.core import setup

setup(name='stockazzo',
      version='alpha',
      description='A library to simulate a trading environment',
      author='Federico Caprini',
      author_email='federico.caprini@protonmail.com',
      url='github.com/yrenum/stockazzo',
      packages=['stockazzo'],
      requires=["stockholm",
                "gymnasium",
                "pandas",
                "numpy",
                "dask",
                "pandas-ta"],
      )
