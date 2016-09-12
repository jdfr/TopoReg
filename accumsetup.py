from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from collections import defaultdict

#run from the command line as: python accumsetup.py build_ext --inplace

extensions = [
    Extension("accum", ["accum.pyx"],
        include_dirs = [numpy.get_include()],
        #libraries = [...],
        #library_dirs = [...]
        ),
]
    

setup(
    name = "Fast numpy accumulator logic",
    ext_modules = cythonize(extensions),
)