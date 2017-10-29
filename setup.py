from distutils.core import setup
from Cython.Build import cythonize

setup(name="train", ext_modules=cythonize('train.pyx'),)

