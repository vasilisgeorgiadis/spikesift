from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension

extensions = [
    Extension("spikesift._cython.alignment", ["spikesift/_cython/alignment.pyx"], include_dirs=[np.get_include()]),
    Extension("spikesift._cython.clustering", ["spikesift/_cython/clustering.pyx"], include_dirs=[np.get_include()]),
    Extension("spikesift._cython.detection", ["spikesift/_cython/detection.pyx"], include_dirs=[np.get_include()]),
    Extension("spikesift._cython.preprocessing", ["spikesift/_cython/preprocessing.pyx"], include_dirs=[np.get_include()]),
    Extension("spikesift._cython.subtraction", ["spikesift/_cython/subtraction.pyx"], include_dirs=[np.get_include()]),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
)
