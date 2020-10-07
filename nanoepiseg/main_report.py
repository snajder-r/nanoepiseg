from typing import IO

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from nanoepitools.nanopolish_container import MetcallH5Container
from nanoepitools.plotting.plot_methylation_profile import plot_met_profile


def read_regions_tsv(regions_tsv_fn: IO) -> pd.DataFrame:
    return pd.read_csv(regions_tsv_fn, sep="\t", dtype={"chromosome": str})


def report(h5_fn: IO, chunk_size: int, regions_tsv_fn: IO, output_pdf_fn: IO):

    regions = read_regions_tsv(regions_tsv_fn)

    with MetcallH5Container(h5_fn, chunk_size=chunk_size) as h5_fp, PdfPages(
        output_pdf_fn
    ) as out_pdf:
        for idx, region in regions.iterrows():
            if idx > 10:
                break
            chrom_container = h5_fp[region["chromosome"]]
            start = region["start"] - 5000
            end = region["end"] + 5000
            values_container = chrom_container.get_values_in_range(start, end)
            met_container = values_container.to_sparse_methylation_matrix("pycometh_rg")

            plt.figure()
            plot_met_profile(
                met_container.met_matrix.todense(), samples=met_container.read_samples
            )
            out_pdf.savefig()
            plt.close()
