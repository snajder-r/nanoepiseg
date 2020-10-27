import argparse
from typing import IO, Iterable

from multiprocessing import Pool
import pandas as pd
from meth5.meth5_wrapper import MetH5File


def read_readgroups(readgroups_file: IO):
    """
    Reads file that assigns read to read groups (such as haplotypes,
    samples, clusters, etc)
    :param readgroups_file: path to the tab-separated file
    :return: pandas dataframe with columns "read_name", "group" and "group_set"
    """
    # Loading
    try:
        read_groups = pd.read_csv(
            readgroups_file,
            sep="\t",
            header=0,
            dtype={"read_name": str, "group": int, "group_set": "category"},
        )
    except Exception as e:
        logging.error("Unable to read read groups file", e)
        raise e

    # Validation
    if len(read_groups.columns) == 2:
        should_colnames = ["read_name", "group"]
    elif len(read_groups.columns) == 3:
        should_colnames = ["read_name", "group", "group_set"]
    else:
        logging.error(
            "Invalid number of columns in read groups file (should be 2 or 3)"
        )
        sys.exit(1)

    if not all([col in read_groups.columns for col in should_colnames]):
        logging.error(
            "Invalid column names in read groups file (should be %s)"
            % should_colnames.join(", ")
        )
        sys.exit(1)

    # Finished validation, now add group_set column if not present
    if "group_set" not in read_groups.columns:
        read_groups["group_set"] = 1
    return read_groups


def chunked_segmentation(
    h5file: IO,
    chunksize: int,
    chunks: Iterable[int],
    readgroups_file: IO,
    workers: int,
    include_nogroup: bool,
):
    pool = Pool(workers)
    print(h5file)

    print("Reading read groups")
    read_groups = read_readgroups(readgroups_file)
    if not include_nogroup:
        read_groups = read_groups.loc[read_groups["group"] != -1]

    read_groups = read_groups.groupby("group_set")
