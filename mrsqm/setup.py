#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Thach Le Nguyen"

from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    name = "mrsqm",
    version = "0.0.1",
    #packages = ["mrsqm"],
    ext_modules = cythonize(Extension(
           name="mrsqm",                                # the extension name
           sources=["mrsqm_wrapper.pyx"], # the Cython source and
                                                  # additional C++ source files
           language="c++",                        # generate and compile C++ code
      )))


