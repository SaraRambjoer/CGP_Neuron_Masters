from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Neuron Search App',
    ext_modules=cythonize("*.pyx"),
    zip_safe=False,
)