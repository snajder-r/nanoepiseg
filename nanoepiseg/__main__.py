import argparse

import nanoepiseg.main_chunked
import nanoepiseg.main_multi
import nanoepiseg.main_extract_haplotype_ids
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
        "chunksize used in creating of the h5 files",
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
    chunked_args.add_argument(
        "--readgroups",
        metavar="readgroups",
        required=True,
        type=str,
        help="File containing read to readgroup assignment",
    )

    chunked_args.add_argument(
        "--include_nogroup",
        action="store_true",
        default=False,
        dest="include_nogroup",
        help="Include reads with group -1 in all windows",
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

    extract_haplotype_ids_args = subparsers.add_parser(
        "extract_haplotype_ids",
        description="Extract phase set and haplotype id from bam file",
    )
    extract_haplotype_ids_args.set_defaults(
        func=nanoepiseg.main_extract_haplotype_ids.extract_haplotype_ids
    )

    extract_haplotype_ids_args.add_argument(
        "--bam",
        metavar="bam",
        type=str,
        required=True,
        help="BAM File containing read group annotation",
    )

    extract_haplotype_ids_args.add_argument(
        "--output",
        metavar="output",
        type=str,
        required=True,
        help="Output tsv file",
    )
    extract_haplotype_ids_args.add_argument(
        "--include_unphased",
        action="store_true",
        default=False,
        dest="include_unphased",
        help="Also include reads that have no HP tag (will be stored as -1/-1 ps/hp)",
    )
    extract_haplotype_ids_args.add_argument(
        "--chroms",
        metavar="chromosomes",
        type=str,
        required=False,
        nargs="*",
        help="Chromosomes to consider (default is all)",
    )

    report_args = subparsers.add_parser("report",
        description="Plot methylation profile for regions", )
    report_args.set_defaults(func=nanoepiseg.main_report.report)

    report_args.add_argument("--regions_tsv_fn", metavar="regions_tsv_fn", type=str,
        required=True, help="TSV output of pycometh report", )
    report_args.add_argument("--gene_expression_file", metavar="regions_tsv_fn", type=str,
        required=True, help="TSV output of pycometh report", )
    report_args.add_argument("--h5_fns", metavar="h5_fns", type=str, nargs="+",
        required=True,
        help="H5 files containing Nanopolish methylation calls, one per sample", )
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
