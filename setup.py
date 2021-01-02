"""Optionally install benchmarks as a module"""
#!/usr/bin/env python

import setuptools

setuptools.setup(name="video_reading_benchmarks",
                 version="X",
                 description="Benchmarking video reading",
                 author="Benjamin Lowe",
                 python_requires=">=3.7",
                 packages=setuptools.find_packages(),
                )
