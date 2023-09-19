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
    version="0.0.4",
    author='Thach Le Nguyen',
    author_email='thalng@protonmail.com',
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.18",
        "pandas>=1.0.3",
        "scikit-learn >= 0.22",
        "pandas>=1.0.3",
    ],
    packages=find_packages(where='src'),
    package_dir={
        '': 'src'
    },
    description='MrSQM: Fast Time Series Classification with Symbolic Representations',
    long_description=open('README.md').read(),
    ext_modules=cythonize([cstuff]),
)