"""
Minimal setup.py for building Cython extensions.

Package metadata is defined in pyproject.toml.
This file is only needed for compiling the optional C extensions.

Usage:
    pip install -e ".[dev]"              # Install without C extensions
    python setup.py build_ext --inplace  # Build C extensions in-place
"""

import sys
import numpy as np
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

extra_compile_args = ['-O3']
libraries = ['m']
if sys.platform == 'win32':
    extra_compile_args = []
    libraries = []

ext_modules = [
    Extension(
        'MTfit.probability.cprobability',
        sources=['src/MTfit/probability/cprobability.pyx'],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
        optional=True,
    ),
    Extension(
        'MTfit.convert.cmoment_tensor_conversion',
        sources=['src/MTfit/convert/cmoment_tensor_conversion.pyx'],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
        optional=True,
    ),
    Extension(
        'MTfit.extensions.cscatangle',
        sources=['src/MTfit/extensions/cscatangle.pyx'],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
        optional=True,
    ),
    Extension(
        'MTfit.algorithms.cmarkov_chain_monte_carlo',
        sources=['src/MTfit/algorithms/cmarkov_chain_monte_carlo.pyx'],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()],
        optional=True,
    ),
]

if HAS_CYTHON:
    ext_modules = cythonize(ext_modules, language_level='3')

setup(
    ext_modules=ext_modules,
)
