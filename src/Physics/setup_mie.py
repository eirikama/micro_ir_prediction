import numpy as np
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import setup

ext = Extension(
    name="mie",
    sources=["mie.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    extra_link_args=["-fopenmp"],
    libraries=["m"],
    language="c++",
)

setup(
    name="mie",
    ext_modules=cythonize(
        ext,
        language_level=3,
        annotate=False,
    ),
)
