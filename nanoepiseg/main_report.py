from typing import IO, List
from pathlib import Path

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from meth5.meth5_wrapper import MetH5File
from nanoepitools.plotting.plot_methylation_profile import plot_met_profile


def read_regions_tsv(regions_tsv_fn: IO) -> pd.DataFrame:
    return pd.read_csv(regions_tsv_fn, sep="\t", dtype={"chromosome": str})


def report(h5_fns: List[IO], chunk_size: int, regions_tsv_fn: IO, output_pdf_fn: IO, gene_expression_file: IO):
    matplotlib.use("Agg")

    regions = read_regions_tsv(regions_tsv_fn)
    regions = regions.iloc[:10]

    sample_names = [Path(h5_fn).stem for h5_fn in h5_fns]
    sample_h5_fp = [
        MetH5File(h5_fn, chunk_size=chunk_size) for h5_fn in h5_fns
    ]
    
    try:
        output_count = 0
        with PdfPages(output_pdf_fn) as out_pdf:
            for idx, region in regions.iterrows():
                chrom = region["chromosome"]
                start = region["start"] - 3000
                end = region["end"] + 3000

                plt.figure(figsize=(10, 18), dpi=300)

                med_llr_diff = []
                met_containers = []

                for i, (h5_fn, h5_fp) in enumerate(zip(sample_names, sample_h5_fp)):
                    chrom_container = h5_fp[chrom]
                    values_container = chrom_container.get_values_in_range(start, end)
                    met_container = values_container.to_sparse_methylation_matrix(
                        "pycometh_rg"
                    )
                    met_dense = np.array(met_container.met_matrix.todense())
                    met_dense[met_dense == 0] = np.nan

                    read_samples = np.array(met_container.read_samples)
                    med_0 = np.nanmedian(met_dense[read_samples == 1,:], axis=0)
                    med_1 = np.nanmedian(met_dense[read_samples == 2,:], axis=0)
                    
                    med_llr_diff.append(np.abs(med_0 - med_1))

                    met_containers.append(met_container)

                # Test for sample differential met
                stat, pval = scipy.stats.kruskal(*med_llr_diff)
                
                if pval < 0.05:
                    output_count += 1

                    for i in range(len(met_containers)):
                        met_container = met_containers[i]
                        sample = sample_names[i]

                        plt.subplot(len(h5_fns), 1, i + 1)
                        plt.title("%s - %s %d %d" % (sample, chrom, start, end))
                        plot_met_profile(
                            np.array(met_container.met_matrix.todense()),
                            samples=np.array(met_container.read_samples),
                            site_genomic_pos=np.array(met_container.genomic_coord),
                            site_genomic_pos_end=np.array(met_container.genomic_coord_end),
                            sample_order=[-1,1,2],
                            sample_colors={-1:'w', 1:'cyan', 2:'magenta'}
                            
                        )

                    out_pdf.savefig()

                plt.close()
    finally:
        for h5_fp in sample_h5_fp:
            h5_fp.close()
