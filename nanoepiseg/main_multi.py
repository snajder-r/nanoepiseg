import sys
import argparse
from pathlib import Path
from typing import IO, List, Tuple, Dict

from multiprocessing import Pool
from meth5.meth5_wrapper import (
    MetH5File,
    MethlyationValuesContainer,
    ChromosomeContainer,
    create_sparse_matrix_from_samples,
)

'''

def segmentation_worker(window_met, rn_hp_dict, n_segments):
    dense_met_llr = np.array(window_met.met_matrix.todense())
    dense_met = dense_met_llr.copy()
    met_prob = net_m.llr_to_p(dense_met[dense_met != 0])
    # We need to ensure some uncertainty, since otherwise likelihoods explode
    met_prob = np.clip(met_prob, np.exp(-16), 1 - np.exp(-16))
    dense_met[dense_met != 0] = met_prob
    dense_met[dense_met == 0] = -1

    if len(dense_met.shape) != 2:
        # Not sure if this ever happens - but just in case the array
        # collapses if too much has been filtered out
        raise ValueError("Input matrix has wrong dimensions")
    if dense_met.shape[0] == 0 or dense_met.shape[1] < 10:
        raise ValueError("Empty input matrix")

    # Since haplotypes might be arbitrary integers, but the
    # segmentation HMM needs integers starting from 0, we translate
    # haplotype ids here to integers starting from 0
    window_hp = [rn_hp_dict[read] for read in window_met.read_names]
    hp_index_dict = {hp: i for i, hp in enumerate(sorted(list(set(window_hp))))}
    window_hp = np.array([hp_index_dict[hp] for hp in window_hp])

    num_samples = len(set(window_hp))

    use_samples_in_hmm = True
    if use_samples_in_hmm:
        emission_lik = BernoulliPosterior(num_samples, n_segments, prior_a=None)
        hmm = SegmentationHMM(n_segments, 0.1, 0.8, emission_lik)
        hmm.baum_welch(dense_met, samples=window_hp)
    else:
        emission_lik = BernoulliPosterior(len(window_hp), n_segments, prior_a=None)
        hmm = SegmentationHMM(n_segments, 0.1, 0.8, emission_lik)
        hmm.baum_welch(dense_met)
    segments_p = emission_lik.segment_p
    segmentation, _ = hmm.viterbi(dense_met, window_hp)
    segmentation = nes_pp.cleanup_segmentation(segments_p, segmentation)
    return segmentation, segments_p, hp_index_dict, dense_met_llr, window_hp


class SegmentationCallback:
    def __init__(
        self,
        sw: SegmentsWriterCSV,
        pa: PlotArchiver,
        window_met,
        ps_id: int,
        chrom: str,
    ):
        self.sw = sw
        self.pa = pa
        self.window_met = window_met
        self.ps_id = ps_id
        self.chrom = chrom

    def segmentation_callback(self, result):
        segmentation, segments_p, hp_index_dict, dense_met_llr, window_hp = result
        print("plotting")
        plt.figure(figsize=(10, 8))
        title_format = "{chrom}:{start}-{end}, Phase Set {ps}"
        start = self.window_met.get_genomic_region()[0]
        end = self.window_met.get_genomic_region()[1]
        plt.title(
            title_format.format(chrom=self.chrom, start=start, end=end, ps=self.ps_id)
        )

        """We need to translate haplotype id (from whatshap: 1 and 2) into what
        the segmentation algorithm knows as the sample index (0 and 1 in
        this case). This also needs to consider that sometimes there is
        only a single haplotype in a window or even the entire phase set,
        which is why we need to check this when constructing the color
        dict"""
        plot_hpindex_colors = {
            hp_index_dict[x]: plot_hp_colors[x]
            for x in plot_hp_colors.keys()
            if x in hp_index_dict.keys()
        }

        net_pm.plot_met_profile(
            dense_met_llr,
            window_hp,
            sorted(list(set(window_hp))),
            plot_hpindex_colors,
            segment=segmentation,
        )
        self.pa.savefig()

        hp_order = sorted(list(hp_index_dict.keys()))
        self.sw.write_segments_llr(
            dense_met_llr,
            segmentation,
            segments_p,
            self.window_met.genomic_coord,
            window_hp,
            self.ps_id,
            hp_index_dict,
            hp_order,
        )

    def error_callback(self, arg):
        start = self.window_met.get_genomic_region()[0]
        end = self.window_met.get_genomic_region()[1]
        print(
            "Warning: Failed for phase set {ps}, window {start}-{"
            "end}".format(ps=self.ps_id, start=start, end=end)
        )
        print(arg)


def segment_phase_set(
    ps: PhaseSet,
    unphased_read_names: List[str],
    sparse_met: net_npc.SparseMethylationMatrixContainer,
    pa: PlotArchiver,
    sw: SegmentsWriterCSV,
    pool: Pool,
    n_segments: int,
):
    ps_id = ps.ps_id

    rn_hp_dict = get_readname_haplotype_dict(ps)
    for read in unphased_read_names:
        rn_hp_dict[read] = 999

    sparse_met = sparse_met.get_submatrix_from_read_names(list(rn_hp_dict.keys()))
    if sparse_met.shape[0] == 0:
        print(
            "Warning: Phase Set %d has %d phased reads but no methylation "
            "calls" % (ps_id, len(ps.read_mappings.keys()))
        )
        return

    winlen = 250
    met_sites = sparse_met.met_matrix.shape[1]
    async_results = []
    for start in range(0, met_sites, winlen):
        print(start)
        window_met = sparse_met.get_submatrix(start, start + winlen)

        callback = SegmentationCallback(sw, pa, window_met, ps_id, ps.chrom)
        set_hp = {rn_hp_dict[read] for read in window_met.read_names}
        if len(set_hp) < 2:
            # No point in segmenting a window that has only reads for a single haplotype
            continue
        ar = pool.apply_async(
            segmentation_worker,
            args=(window_met, rn_hp_dict, n_segments),
            callback=callback.segmentation_callback,
            error_callback=callback.error_callback,
        )
        async_results.append(ar)

    for i, res in enumerate(async_results):
        res.wait(timeout=1800)
        print("%d from %d workers completed" % (i, len(async_results)))'''


def segment(
    sample_h5path: Dict[str, Path],
    genomic_range: Tuple[str, int, int],
    pool: Pool,
    chunk_size: int,
):
    # Open HDF5 files
    sample_h5f: Dict[str, MetH5File] = {
        s: MetH5File(p, "r", chunk_size=chunk_size)
        for s, p in sample_h5path.items()
    }
    # Create dictionary of MethlyationValuesContainer objects for each sample
    sample_met_llrs: Dict[str, MethlyationValuesContainer] = {
        s: f[genomic_range[0]].get_values_in_range(genomic_range[1], genomic_range[2])
        if f[genomic_range[0]] is not None
        else None
        for s, f in sample_h5f.items()
    }

    sparse_met_matrix = create_sparse_matrix_from_samples(sample_met_llrs)
    print(sparse_met_matrix.shape)


def multifile_segmentation(
    h5files: List[str],
    genomic_range: str,
    sample_names: List[str],
    workers: int,
    chunk_size: int,
):

    pool = Pool(workers)
    print(h5files)

    # Parse and validate arguments
    genomic_range = genomic_range.split(":")
    chromosome = genomic_range[0]
    if len(genomic_range) > 1:
        genomic_range = genomic_range[1].split("-")
        try:
            start = int(genomic_range[0].strip())
            end = int(genomic_range[1].strip())
        except ValueError:
            raise ValueError(
                "Invalid format for genomic region. Should be chrom[:start-end]. Start and end must be numeric."
            )
        except IndexError:
            raise ValueError(
                "Invalid format for genomic region. Should be chrom[:start-end]. Both start and end must be provided together and separated by a hyphen."
            )
    else:
        start = 0
        end = sys.maxsize

    # Validate h5 files
    h5files = [Path(f) for f in h5files]
    for f in h5files:
        if not f.exists() or not f.is_file():
            raise ValueError("Invalid HDF5 file, cannot read: \n  %s" % str(f))

    if sample_names is not None:
        if len(sample_names) != len(h5files):
            raise ValueError(
                "If sample names are provided, the number of sample names must be equal to the number of HDF5 files"
            )
    else:
        # Sample names are none, infer from filenames
        sample_names = [f.stem for f in h5files]

    segment(dict(zip(sample_names, h5files)), (chromosome, start, end), pool, chunk_size)
