import time
from pathlib import Path
from typing import IO, Iterable, List, Optional
from multiprocessing import Queue, Process
import argparse
import logging

import tqdm
import pandas as pd
import numpy as np
from meth5.meth5 import MetH5File
from meth5.sparse_matrix import SparseMethylationMatrixContainer

from nanoepiseg.emissions import BernoulliPosterior
from nanoepiseg.hmm import SegmentationHMM
from nanoepiseg.postprocessing import cleanup_segmentation
from nanoepiseg.segments_csv_io import SegmentsWriterBED
from nanoepiseg.math import llr_to_p


def worker_segment(input_queue: Queue, output_queue: Queue, chromosome: str, max_segments_per_window: int):
    import warnings
    
    warnings.filterwarnings("ignore")
    
    while True:
        job = input_queue.get()
        if job is None:
            break
        
        sparse_matrix, fraction = job
        
        llrs = np.array(sparse_matrix.met_matrix.todense())
        obs = llr_to_p(llrs)
        samples = sparse_matrix.read_samples
        
        unique_samples = list(set(samples))
        
        id_sample_dict = {i: s for i, s in enumerate(unique_samples)}
        sample_id_dict = {v: k for k, v in id_sample_dict.items()}
        
        sample_ids = [sample_id_dict[s] for s in samples]
        
        emission_lik = BernoulliPosterior(len(unique_samples), max_segments_per_window, prior_a=None)
        hmm = SegmentationHMM(
            max_segments=max_segments_per_window, t_stay=0.1, t_move=0.8, e_fn=emission_lik, eps=np.exp(-512)
        )
        segment_p, posterior = hmm.baum_welch(obs, tol=np.exp(-8), samples=sample_ids)
        
        segmentation, _ = hmm.MAP(posterior)
        
        segment_p_array = np.concatenate([v[np.newaxis, :] for v in segment_p.values()], axis=0)
        segmentation = cleanup_segmentation(segment_p_array, segmentation, min_parameter_diff=0.2)
        
        used_segments = list(set(segmentation))
        
        result_tuple = (llrs, segmentation, sparse_matrix.genomic_coord, sparse_matrix.read_samples)
        
        output_queue.put((result_tuple, fraction))


def worker_output(
    output_queue: Queue, out_tsv_file: IO, chromosome: str, read_groups_key: str, print_diff_met: bool, quiet: bool
):
    writer = SegmentsWriterBED(out_tsv_file, chromosome)
    with tqdm.tqdm(total=100) as pbar:
        while True:
            res = output_queue.get()
            if res is None:
                break
            
            seg_result, fraction = res
            llrs, segments, genomic_locations, samples = seg_result
            
            if read_groups_key is None:
                samples = None
            
            writer.write_segments_llr(llrs, segments, genomic_locations, samples, compute_diffmet=print_diff_met)
            pbar.update(fraction)
        pbar.n = 100
        pbar.refresh()


def worker_reader(
    m5file: Path,
    chunk_size: int,
    chromosome: str,
    window_size: int,
    input_queue: Queue,
    chunks: List[int],
    progress_per_chunk: float,
    read_groups_key: str,
):
    with MetH5File(m5file, "r", chunk_size=chunk_size) as m5:
        chrom_container = m5[chromosome]
        
        for chunk in chunks:
            values_container = chrom_container.get_chunk(chunk)
            met_matrix: SparseMethylationMatrixContainer = values_container.to_sparse_methylation_matrix(
                read_read_names=False, read_groups_key=read_groups_key
            )
            if read_groups_key is None:
                met_matrix.read_samples = met_matrix.read_names
            total_sites = met_matrix.met_matrix.shape[0]
            num_windows = (total_sites // window_size) + 1
            progress_per_window = progress_per_chunk / num_windows
            for window_start in range(0, total_sites, window_size):
                window_end = window_start + window_size
                # logging.debug(f"Submitting window {window_start}-{window_end}")
                sub_matrix = met_matrix.get_submatrix(window_start, window_end)
                input_queue.put((sub_matrix, progress_per_window))


def validate_chromosome_selection(m5file: Path, chromosome: str, chunk_size: int):
    with MetH5File(m5file, "r", chunk_size=chunk_size) as m5:
        if chromosome not in m5.get_chromosomes():
            raise ValueError(f"Chromosome {chromosome} not found in m5 file.")


def validate_chunk_selection(m5file: Path, chromosome: str, chunk_size: int, chunks: List[int]):
    with MetH5File(m5file, "r", chunk_size=chunk_size) as m5:
        num_chunks = m5[chromosome].get_number_of_chunks()
        if max(chunks) >= m5[chromosome].get_number_of_chunks():
            raise ValueError(f"Chunk {max(chunks)} not in chromosome. Must be in range {0}-{num_chunks-1}")


def main(
    m5file: Path,
    chromosome: str,
    chunk_size: int,
    chunks: Optional[Iterable[int]],
    workers: int,
    reader_workers: int,
    quiet: bool,
    out_tsv: IO,
    window_size: int,
    max_segments_per_window: int,
    read_groups_key: str,
    print_diff_met: bool,
):
    # TODO expose
    input_queue = Queue(maxsize=workers * 5)
    output_queue = Queue(maxsize=workers * 100)
    
    validate_chromosome_selection(m5file, chromosome, chunk_size)
    
    if chunks is None:
        # No chunks have been provided, take all
        with MetH5File(m5file, mode="r", chunk_size=chunk_size) as f:
            chunks = list(range(f[chromosome].get_number_of_chunks()))
    else:
        # flatten chunk list, since we allow a list of chunks or a list of chunk ranges
        # (which are converted to lists in parsing)
        chunks = [chunk for subchunks in chunks for chunk in ([subchunks] if isinstance(subchunks, int) else subchunks)]
    
    validate_chunk_selection(m5file, chromosome, chunk_size, chunks)
    
    # sort and make unique
    chunks = sorted(list(set(chunks)))
    progress_per_chunk = 100 / len(chunks)
    
    segmentation_processes = [
        Process(target=worker_segment, args=(input_queue, output_queue, chromosome, max_segments_per_window))
    ]
    for p in segmentation_processes:
        p.start()
    
    reader_workers = min(reader_workers, len(chunks))
    chunk_per_process = np.array_split(chunks, reader_workers)
    reader_processes = [
        Process(
            target=worker_reader,
            args=(
                m5file,
                chunk_size,
                chromosome,
                window_size,
                input_queue,
                p_chunks,
                progress_per_chunk,
                read_groups_key,
            ),
        )
        for p_chunks in chunk_per_process
    ]
    for p in reader_processes:
        p.start()
    
    output_process = Process(
        target=worker_output, args=(output_queue, out_tsv, chromosome, read_groups_key, print_diff_met, quiet)
    )
    output_process.start()
    
    for p in reader_processes:
        p.join()
    
    # Deal poison pills to segmentation workers
    for p in segmentation_processes:
        input_queue.put(None)
    
    for p in segmentation_processes:
        p.join()
    
    # Deal poison pill to segmentation workers
    output_queue.put(None)
    output_process.join()
