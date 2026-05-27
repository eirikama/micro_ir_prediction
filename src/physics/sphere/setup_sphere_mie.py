import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

os.environ["CC"] = "gcc"

ext = Extension(
    name="sphere_mie",
    sources=["sphere_mie.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native", "-ffast-math", "-fopenmp", "-funroll-loops"],
    extra_link_args=["-fopenmp"],
    libraries=["m"],
    language="c++",
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="sphere_mie",
    ext_modules=cythonize(
        ext,
        language_level=3,
        annotate=False,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
        },
    ),
)
