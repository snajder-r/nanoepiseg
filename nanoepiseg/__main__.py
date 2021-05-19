import argparse
from pathlib import Path
import logging

from meth5.meth5 import MetH5File

import nanoepiseg.main_chunked
import nanoepiseg.main_list_chunks


def argtype_M5File(value):
    try:
        MetH5File(value, "r").get_chromosomes()
    except:
        raise argparse.ArgumentTypeError(f"Failed to read '{value}'. Is it a valid MetH5 file?")
    return Path(value)

def argtype_chunks(value : str):
    try:
        if "-" in value:
            split = value.split("-")
            if len(split) != 2:
                raise argparse.ArgumentTypeError(f"Argument 'chunks' must be space separated list of integers, or ranges in format 'from-to'. Can't parse '{value}'")
            return list(range(int(split[0]), int(split[1])))
        else:
            return [int(value)]
    except:
        raise argparse.ArgumentTypeError(
            f"Argument 'chunks' must be space separated list of integers, or ranges in format 'from-to'. Can't parse '{value}'")

def main():
    parser = argparse.ArgumentParser(
        description="HMM based de-novo segmentation of methylation from Nanopolish methylation calls)"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=int(1e6),
        help="Number of llrs per chunk - for best "
        "performance, should be a multiple of the "
        "chunksize used in creating of the h5 files",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize log output",
    )
    
    subparsers = parser.add_subparsers(description="Subcommand: ", dest="subcommand")
    subparsers.required = True
    
    sc_args = subparsers.add_parser("list_chunks", description="List the number of chunks per chromosome")
    
    sc_args.add_argument(
        "--m5file",
        required=True,
        type=argtype_M5File,
        help="MetH5 file containing methylation calls",
    )
    sc_args.set_defaults(func=nanoepiseg.main_list_chunks.main)
    
    sc_args = subparsers.add_parser(
        "segment_h5",
        description="Segment a single HDF5 file",
    )
    sc_args.set_defaults(func=nanoepiseg.main_chunked.main)
    
    sc_args.add_argument(
        "--chunks",
        type=argtype_chunks,
        nargs="+",
        required=False,
        help="Chunk ids within chromosome in hdf5 file. If none provided, all chunks will be processed. Can also provide ranges as 'start-end'",
    )
    
    sc_args.add_argument(
        "--reader_workers",
        type=int,
        help="Number of processes for reading the input file. Recommended are at least half the number as there are worker processes",
        default=2,
    )

    sc_args.add_argument("--workers", type=int, help="Number of worker processes", default=4, )

    sc_args.add_argument(
        "--m5file",
        required=True,
        type=argtype_M5File,
        help="MetH5 file containing methylation calls",
    )
    
    sc_args.add_argument("--chromosome", type=str, required=True, help="The chromosome or contig you want to segment")
    
    sc_args.add_argument(
        "--out_tsv",
        required=True,
        type=Path,
        help="Tab-separated output file containing coordinates of segments",
    )
    
    sc_args.add_argument(
        "--window_size", type=int, required=False, default=300, help="Window size for segmentation algorithm (number of CpGs)."
    )
    sc_args.add_argument(
        "--max_segments_per_window",
        type=int,
        required=False,
        default=15,
        help="Maximum number of segments per window (greatly affects performance)",
    )
    
    sc_args.add_argument(
        "--read_groups_key",
        type=str,
        required=False,
        help="If the H5 files is tagged with read groups, provide the read group key here. If done so, methylation rates will be modeled per read group instead of per read.",
    )

    sc_args.add_argument("--print_diff_met", action="store_true", help="Compute differential methylation p-values between read groups (i.e. samples/haplotypes) and include that information int he output file", )

    args = parser.parse_args()
    args_dict = vars(args)
    # Remove arguments that the subcommand doesn't take
    subcommand = args.func
    del args_dict["subcommand"]
    del args_dict["func"]
    
    try:
        subcommand(**args_dict)
    except ValueError as e:
        logging.error(str(e))
        
    
