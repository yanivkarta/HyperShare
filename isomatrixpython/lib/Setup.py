'''
This file is used to setup the library.
for example, the library is compiled and the shared object is loaded. 
The functions and classes are defined here.

'''


from distutils.core import setup, Extension

module = Extension('libisolation_mat_python',
                    sources = ['FastMatrixForest.cpp', 'PythonFastMatrixForest.cpp'],
                    extra_compile_args=['-std=c++11'],
                    extra_link_args=['-std=c++11'])

setup (name = 'libisolation_mat_python',
         version = '1.0',
         description = 'This is a package for the isolation matrix algorithm',
         ext_modules = [module])


# Path: lib/FastMatrixForest.cpp
