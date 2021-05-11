#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

# Long description from README file
with open("README.md", "r") as fh:
    long_description = fh.read()

# Collect info in a dictionary for setup.py
setup(
    name="nanoepiseg",
    description="Methylome segmentation algorithm using a changepoint detection HMM",
    version="0.0.2.dev1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snajder-r/nanoepiseg",
    author="Rene Snajder",
    author_email="r.snajder@dkfz-heidelberg.de",
    license="MIT",
    python_requires=">=3.7",
    classifiers=["Development Status :: 3 - Alpha", "Intended Audience :: Science/Research", "Topic :: Scientific/Engineering :: Bio-Informatics", "License :: OSI Approved :: MIT License", "Programming Language :: Python :: 3"],
    install_requires=["numpy>=1.19.2", "scipy==1.4.1", "pandas>=1.1.3", "meth5>=0.2.6"],
    packages=["nanoepiseg"],
    package_dir={"nanoepiseg": "nanoepiseg"},
    entry_points={"console_scripts": ["nanoepiseg=nanoepiseg.__main__:main"]},
)
