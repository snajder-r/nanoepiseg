from pathlib import Path
from meth5.meth5 import MetH5File


def main(m5file:Path, chunk_size:int, quiet:bool):
    with MetH5File(m5file, "r", chunk_size=chunk_size) as f:
        for chrom in f.get_chromosomes():
            print(f"{chrom}: {f[chrom].get_number_of_chunks()}")
