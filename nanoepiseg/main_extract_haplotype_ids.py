import argparse
from typing import IO, Iterable
import json

from multiprocessing import Pool
from nanoepitools.haplotype_tools import extract_read_haplotype_assignment


def extract_haplotype_ids(
    bam: IO, output: IO, chroms: Iterable[str], include_unphased: bool
):
    print("Reading bam file")
    if len(chroms) == 1 and chroms[0] == "all":
        chroms = None
    ps_collection = extract_read_haplotype_assignment(
        bam, chroms=chroms, return_unphased=include_unphased
    )
    print("Writing output")

    out_line = "{read_name}\t{hp}\t{ps}\n"

    with open(output, "w") as out_f:
        if include_unphased:
            for k, reads in ps_collection.unphased.items():
                for read in reads:
                    out_f.write(out_line.format(read_name=read.read_name, hp=-1, ps=-1))

        for ps_id, ps in ps_collection.phased.items():
            for chrom_reads in ps.read_mappings.values():
                for read in chrom_reads:
                    out_f.write(
                        out_line.format(
                            read_name=read.read_name, hp=read.haplotype, ps=ps_id
                        )
                    )
