#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Thach Le Nguyen"

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

cstuff = Extension('mrsqm.mrsqm_wrapper',
                   language='c++',
                   sources=["src/mrsqm/mrsqm_wrapper.pyx","src/mrsqm/sfa/MFT.cpp","src/mrsqm/sfa/DFT.cpp","src/mrsqm/sfa/SFA.cpp","src/mrsqm/sfa/TimeSeries.cpp"],
                   extra_compile_args=["-Wall", "-Ofast", "-g", "-std=c++11", "-ffast-math"],
                   extra_link_args=["-lfftw3", "-lm", "-L/opt/local/lib"],           
                   include_dirs=['src/mrsqm'])

setup(
    name='mrsqm',
    version="0.0.2",
    author='Thach Le Nguyen',
    author_email='thalng@protonmail.com',
    setup_requires=[
        'setuptools',  # first version to support pyx in Extension
        'cython',
    ],
    packages=find_packages(where='src'),
    package_dir={
        '': 'src'
    },
    #description='Example python module with cython.',
    #long_description=open('README.md').read(),
    ext_modules=cythonize([cstuff]),
)