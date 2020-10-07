import argparse

import nanoepiseg.main_chunked
import nanoepiseg.main_multi
import nanoepiseg.main_report


def main():
    parser = argparse.ArgumentParser(
        description="HMM based de-novo segmentation of methylation from Nanopolish methylation calls)"
    )

    parser.add_argument(
        "--chunk_size",
        metavar="chunk_size",
        type=int,
        required=True,
        help="Number of llrs per chunk - for best "
        "performance, should be a multiple of the "
        "chunk_size used in creating of the h5 files",
    )

    subparsers = parser.add_subparsers(description="Subcommand: ", dest="subcommand")
    subparsers.required = True

    chunked_args = subparsers.add_parser(
        "single_h5",
        description="Segment a single HDF5 file based on a separate read-to-sample mapping file",
    )
    chunked_args.set_defaults(func=nanoepiseg.main_chunked.chunked_segmentation)

    chunked_args.add_argument(
        "--chunks",
        metavar="chunks",
        type=int,
        nargs="+",
        required=True,
        help="Chunk ids within chromosome in hdf5 file. If more than one hdf5 is "
        "provided, the chunk id is picked from the file with the most chunks, and "
        "the other hdf5 files are mapped based on the genomic coordinates",
    )

    chunked_args.add_argument(
        "--workers",
        metavar="workers",
        type=int,
        help="Number of worker processes",
        default=10,
    )
    chunked_args.add_argument(
        "--h5file",
        metavar="h5file",
        required=True,
        type=str,
        help="H5 file containing Nanopolish methylation calls",
    )

    multi_h5_args = subparsers.add_parser(
        "multi_h5",
        description="Segment multiple H5 files, where each h5 file represents one sample",
    )
    multi_h5_args.set_defaults(func=nanoepiseg.main_multi.multifile_segmentation)

    multi_h5_args.add_argument(
        "--genomic_range",
        metavar="genomic_range",
        type=str,
        required=True,
        help="Genomic range in format chr[:start-end]",
    )
    multi_h5_args.add_argument(
        "--workers",
        metavar="workers",
        type=int,
        help="Number of worker processes",
        default=10,
    )
    multi_h5_args.add_argument(
        "--h5files",
        metavar="h5files",
        nargs="+",
        type=str,
        required=True,
        help="H5 files containing Nanopolish methylation calls, one file per sample",
    )
    multi_h5_args.add_argument(
        "--sample_names",
        metavar="sample_names",
        nargs="*",
        type=str,
        help="Sample names (one per h5 file). If not provided, will be inferred from the filenames",
    )

    report_args = subparsers.add_parser("report",
        description="Plot methylation profile for regions", )
    report_args.set_defaults(func=nanoepiseg.main_report.report)

    report_args.add_argument("--regions_tsv_fn", metavar="regions_tsv_fn", type=str,
        required=True, help="TSV output of pycometh report", )
    report_args.add_argument("--h5_fn", metavar="h5_fn", type=str,
        required=True,
        help="H5 file containing Nanopolish methylation calls", )
    report_args.add_argument("--output_pdf_fn", metavar="output_pdf_fn", type=str,
        required=True,
        help="PDF output file", )

    args = parser.parse_args()
    args_dict = vars(args)
    # Remove arguments that the subcommand doesn't take
    subcommand = args.func
    del args_dict["subcommand"]
    del args_dict["func"]

    subcommand(**args_dict)