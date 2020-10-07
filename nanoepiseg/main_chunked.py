import argparse
from typing import IO, Iterable

from multiprocessing import Pool
from nanoepitools.nanopolish_container import MetcallH5Container

def chunked_segmentation(h5file: IO, chunk_size: int, chunks: Iterable[int], workers: int):
    
    pool = Pool(workers)
    print(h5file)

        
    