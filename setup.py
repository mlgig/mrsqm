__author__ = "Thach Le Nguyen"

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os 

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()

def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

cstuff = Extension('mrsqm.mrsqm_wrapper',
                   language='c++',
                   sources=["src/mrsqm/mrsqm_wrapper.pyx","src/mrsqm/sfa/MFT.cpp","src/mrsqm/sfa/DFT.cpp","src/mrsqm/sfa/SFA.cpp","src/mrsqm/sfa/TimeSeries.cpp"],
                   extra_compile_args=["-Wall", "-Ofast", "-g", "-std=c++11", "-ffast-math"],
                   extra_link_args=["-lfftw3", "-lm", "-L/opt/local/lib"],           
                   include_dirs=['src/mrsqm'])

setup(
    name='mrsqm',
    version=get_version("src/mrsqm/__init__.py"),
    author='Thach Le Nguyen',
    author_email='thalng@protonmail.com',
    python_requires='>=3.7',
    install_requires=[
        "numpy>=1.18",
        "pandas>=1.0.3",
        "scikit-learn >= 0.22",        
    ],
    packages=find_packages(where='src'),
    package_dir={
        '': 'src'
    },
    description='MrSQM: Fast Time Series Classification with Symbolic Representations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=cythonize([cstuff]),
)