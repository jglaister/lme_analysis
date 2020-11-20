#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Setup for lme_analysis package

Author: Jeffrey Glaister
"""
from glob import glob
from setuptools import setup, find_packages

args = dict(
    name='lme_analysis',
    version='0.1',
    description="Leptomengineal enhancement analysis",
    author='Jeffrey Glaister',
    author_email='jeff.glaister@gmail.com',
    url='https://github.com/jglaister/lme_analysis',
    keywords="central vein sign"
)

setup(install_requires=['nipype', 'vtk', 'nibabel', 'numpy', 'sklearn', 'scipy', 'matplotlib'],
      packages=['lme_pipeline'],
      scripts=glob('bin/*'), **args)
