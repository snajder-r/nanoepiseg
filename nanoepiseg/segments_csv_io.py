from typing import IO
from io import StringIO

import numpy as np
import pandas as pd


class SegmentsWriterBED:

    def __init__(self, outfile: IO, chrom: str):
        self.outfile = outfile
        self.chrom = chrom
        self.first = True

    def write_segments_llr(
        self,
        llrs: np.ndarray,
        segments: np.ndarray,
        segment_p: np.ndarray,
        genomic_locations: np.ndarray,
        samples: np.ndarray
    ):
        df = pd.DataFrame(columns=["chrom", "start", "end", "num_sites"])
        df = df.astype({"chrom": str, "start": int, "end": int, "num_sites": int})

        for seg in sorted(list(set(segments))):
            seg_pos = np.arange(llrs.shape[1])[segments == seg]
            start = genomic_locations[seg_pos[0]]
            end = genomic_locations[seg_pos[-1]]

            rowval = {
                "chrom": self.chrom,
                "start": start,
                "end": end,
                "num_sites": (segments == seg).sum()
            }

            for i, hp_a in enumerate(hps):
                for hp_b in hps[i + 1 :]:
                    up, pp = compute_differential_methylation(
                        hp_llrs[hp_a], hp_llrs[hp_b]
                    )
                    rowval["unpaired_pval_%s_vs_%s" % (hp_a, hp_b)] = up
                    rowval["paired_pval_%s_vs_%s" % (hp_a, hp_b)] = pp

            rowval.update({"HP%d_p" % hp: hp_p[hp] for hp in hps})

            def format_llrs(x):
                str_list = ["%.2f" % llr for llr in hp_llrs[x][hp_llrs[x] != 0]]
                return ",".join(str_list)

            rowval.update({"HP%d_llrs" % hp: format_llrs(hp) for hp in hps})

            df = df.append(rowval, ignore_index=True)

        df.to_csv(self.outfile, sep="\t", header=True, index=False)
        # self.first = False


class SegmentsReaderCSV:
    """
    Reads a concatenation of CSV files (with potentially differenct columns) and merges
    them into one dataframe, filling in null for missing values. Accepts multiple header
    lines, but requires them to start with the word "chrom" in order to identify the
    header.
    
    This is used to read a CSV file that has been created in chunks by multiple
    concurrent worker threads.
    """
    def __init__(self, *argc):
        self.inputfiles = argc

    def read(self):
        ret = []
        for inputfile in self.inputfiles:
            with open(inputfile) as f:
                cur_string = None
                while True:
                    line = f.readline().strip()
                    if line.startswith("chrom") or not line:
                        # new header line
                        if cur_string is not None:
                            cur_pd = pd.read_csv(
                                StringIO("\n".join(cur_string)), sep="\t"
                            )
                            ret.append(cur_pd)
                        cur_string = []
                    if not line:
                        break
                    cur_string.append(line)
        ret = pd.concat(ret, sort=False)
        ret = ret.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        return ret