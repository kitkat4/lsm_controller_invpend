from distutils.core import setup
from Cython.Build import cythonize

setup(name="resume", ext_modules=cythonize('resume.pyx'),)

