# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
from setuptools           import setup, find_packages
from distutils.extension import Extension
from Cython.Build         import cythonize

extensions = [
        Extension(
                "cdiv",
                ["cdiv.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'])]


setup(
    name        = "cdiv",
    ext_modules = cythonize(extensions)
)