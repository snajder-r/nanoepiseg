# NanoEpiSeg 1.0.0b3

[![GitHub license](https://img.shields.io/github/license/snajder-r/nanoepiseg)](https://github.com/snajder-r/nanoepiseg/blob/master/LICENSE)
[![Language](https://img.shields.io/badge/Language-Python3.7+-yellow.svg)](https://www.python.org/)
[![Build Status](https://travis-ci.com/snajder-r/nanoepiseg.svg?branch=main)](https://travis-ci.com/snajder-r/nanoepiseg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg?style=flat)](https://github.com/snajder-r/black "Black (modified)")

[![PyPI version](https://badge.fury.io/py/nanoepiseg.svg)](https://badge.fury.io/py/nanoepiseg)
[![PyPI downloads](https://pepy.tech/badge/nanoepiseg)](https://pepy.tech/project/nanoepiseg)
[![Anaconda Version](https://img.shields.io/conda/v/snajder-r/nanoepiseg?color=blue)](https://anaconda.org/snajder-r/nanoepiseg)
[![Anaconda Downloads](https://anaconda.org/snajder-r/nanoepiseg/badges/downloads.svg)](https://anaconda.org/snajder-r/nanoepiseg)

NanoEpiSeg is a tool for *de novo* segmentation of  a methylome from read-level methylation calls (such as Nanopolish).

NanoEpiSeg is currently in development. Please do not hesitate to report bugs or feature requests.

A detailed documentation is in the works, stay tuned!
## Prerequisites

NanoEpiSeg assumes that your methylation calls are stored in [MetH5](http://github.com/snajder-r/meth5format) format.

## Installation

Through pip:

```
pip install nanoepiseg
````

Through anaconda:

```
conda install -c snajder-r nanoepiseg
```

## Usage

NanoEpiSeg is meant to be parallelizable, which is why you would typically call nanoepiseg on a cluster system in parallel.
In order to best accomplish load-balancing, it is done per hdf5 chunk. 

You can list the number of chunks per chromosome via:

```bash
nanoepiseg list_chunks --m5file INPUT_FILE.m5
```

To then perform segmentation for a certain chunk, you can run:

```bash
nanoepiseg segment_h5 --m5file INPUT_FILE.m5 \
    --out_tsv OUTPUT_FILE \ 
    --reader_workers NUM_READER_WORKERS \ 
    --workers NUM_SEGMENTATION_WORKERS \ 
    --chromosome CHROMOSOME \
    --chunks CHUNK1 [CHUNK2 ...]
```

There are further options available. Please check out the help to discover them.


