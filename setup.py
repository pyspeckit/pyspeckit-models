#!/usr/bin/env python

import os
import shutil
import sys
from distutils.core import setup, Command

if 'develop' in sys.argv:
    # use setuptools for develop, but nothing else
    from setuptools import setup

setup(name='pyspeckitmodels',
      version='0.1',
      description='Models for use with pyspeckit',
      author=['Adam Ginsburg','Jordan Mirocha'],
      author_email=['adam.g.ginsburg@gmail.com', 'mirochaj@gmail.com',
          'pyspeckit@gmail.com'], 
      url='https://pyspeckit.bitbucket.org/',
      packages=['pyspeckitmodels',
                'pyspeckitmodels.hydrogen',
                'pyspeckitmodels.support',
                'pyspeckitmodels.h2'],
      package_data={'pyspeckitmodels':['co/*txt','h2/*txt']},
      requires=['matplotlib (>=1.1.0)','numpy (>=1.4.1)','pyspeckit'],
      classifiers=[
                   "Development Status :: 3 - Alpha",
                   "Programming Language :: Python",
                   "License :: OSI Approved :: MIT License",
                  ],
      
     )

