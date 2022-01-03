#!/usr/bin/env python
"""Optionally install benchmarks as a module"""

import setuptools

setuptools.setup(name="video_reading_benchmarks",
                 version="X",
                 description="Benchmarking video reading",
                 author="Benjamin Lowe",
                 python_requires=">=3.7",
                 packages=setuptools.find_packages(),
                )
