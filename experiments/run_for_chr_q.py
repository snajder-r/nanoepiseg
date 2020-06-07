import os

import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import scnmttools.h5utils as h5util
import re
import pandas as pd
import scipy.sparse  as sp

from nanoepiseg.helper import medulloblastoma_project
import nanoepiseg.hmm
import nanoepiseg.helper
import time

from IPython.display import clear_output
import ruptures as rpt
from scipy import signal
import sys

metcall_dir = '/home/r933r/snajder/nanopore/data/medulloblastoma_dna/met_merged/'
chrom = sys.argv[1]
qs = float(sys.argv[2])
qe = float(sys.argv[3])
sfx = sys.argv[4]
#outfile_ml = '/home/r933r/snajder/nanopore/data/medulloblastoma_dna/met_seg/ml_%s_%s.txt'%(chrom,sfx)
#outfile_map = '/home/r933r/snajder/nanopore/data/medulloblastoma_dna/met_seg/map_%s_%s.txt'%(chrom,sfx)
#pdf_ml = '/home/r933r/snajder/nanopore/data/medulloblastoma_dna/met_seg/ml_%s_%s.pdf'%(chrom,sfx)
#pdf_map = '/home/r933r/snajder/nanopore/data/medulloblastoma_dna/met_seg/map_%s_%s.pdf'%(chrom,sfx)

outfile_ml = '/home/r933r/snajder/test/ml_%s_%s.txt'%(chrom,sfx)
outfile_map = '/home/r933r/snajder/test/map_%s_%s.txt'%(chrom,sfx)
pdf_ml = '/home/r933r/snajder/test/ml_%s_%s.pdf'%(chrom,sfx)
pdf_map = '/home/r933r/snajder/test/map_%s_%s.pdf'%(chrom,sfx)


mb = medulloblastoma_project()
q_filter = nanoepiseg.helper.QuantileFilter(qs,qe)

mb.load_pickled_metcalls(metcall_dir, chrom, region_filter=q_filter,filter_bad_reads=False)

mb.build_read_adjacency_matrix()
mb.build_sparse_met_matrix()

pdf_ml = matplotlib.backends.backend_pdf.PdfPages(pdf_ml)
pdf_map = matplotlib.backends.backend_pdf.PdfPages(pdf_map)
ml_f = open(outfile_ml,'w')
map_f = open(outfile_map,'w')

start = 0
ws = 300
st = time.time()


def cleanup_segmentation(segment_p, X):
    newX = X.copy()
    lastx = 0
    for x in sorted(list(set(X))):  
        length = (newX==x).sum()
        if x == newX[-1]:
            cadidate_replace = newX[newX!=x][-1]
        else:
            cadidate_replace = x+1
        absdif = np.abs(segment_p[:,x]-segment_p[:,cadidate_replace]).max()
        if length < 5 or absdif<0.1:
            newX[newX==x] = cadidate_replace
    
    return np.array(newX)

while start < mb.met_matrix.shape[1]:
    if start > 0:
        ct = time.time()
        progress = start / mb.met_matrix.shape[1]
        spp = (ct - st)/start
        srest = spp * (mb.met_matrix.shape[1]-start)
        print("Done %f, took %f sec, hours remaining: %f" % (progress, (ct-st), srest/3600))
        
    end = min(start + ws, mb.met_matrix.shape[1])
    part_matrix = mb.met_matrix[:,start:end]
    part_genomic_pos = mb.unique_sites[start:end]
    read_has_data = np.array(((part_matrix!=0).sum(axis=1)>10)).flatten()
    part_matrix = np.array(part_matrix[read_has_data,:].todense())
    print(part_matrix.shape)
    part_samples = mb.matrix_samples[read_has_data]
    

    eps = np.exp(-512)
    def emission_probs(s, o, segment_p, segment_prior):
        idx = (o!=-1)
        ret_a = (np.log(1-np.exp(segment_p[idx,s])+eps)) + np.log(1-o[idx]) + np.log(0.5)
        ret_b = segment_p[idx,s] + np.log(o[idx]) + np.log(0.5) 
        ret = np.logaddexp(ret_a,ret_b)
        if not segment_prior is None:
            ret += segment_prior[idx,s]
        return ret.sum()


    metrate = 0.5
    part_matrix = np.clip(part_matrix, -64,+64)
    met_prob = 1/(1 + np.exp(-part_matrix)/(metrate) - np.exp(-part_matrix))
    met_prob[part_matrix==0] = -1

    sample_map = np.array([0 if s=='s1' else 1 if s=='s2' else 2 for s in part_samples])
    max_seg=10
    model = nanoepiseg.hmm.SegmentationHMM(max_seg, 0.1,0.8,0.0, eps=np.exp(-(2**9)))

    segment_p, segment_prior, posterior = model.baum_welch(met_prob, 
            e_fn_unsalted=emission_probs, tol=0.001, samples=sample_map)


    def write_region(writer, X):
        for x in sorted(list(set(X))):
            absmax = 0
            absmax_pair = (-1,-1)
            for c in range(1,segment_p.shape[0]):
                curmax = np.abs(np.exp(segment_p[c,x])-np.exp(segment_p[c-1,x]))
                if curmax > absmax:
                    absmax_pair = (c-1,c)
                    absmax = curmax
            rs,rend = np.where(X==x)[0][0],np.where(X==x)[0][-1]
            real_start = mb.unique_sites[start+rs]
            real_end = mb.unique_sites[start+rend]
            writer.write('{chrom}:{start}-{end}\t{first}_vs_{second}\t{diff}\n'.format(chrom=chrom,start=real_start, 
                end=real_end, first=absmax_pair[0], second=absmax_pair[1], diff=absmax))


    X,Z = model.MAP(posterior)
    X = cleanup_segmentation(segment_p, X)
    write_region(map_f,X)

    fig = plt.figure(figsize=(12,6))
    nanoepiseg.helper.plotting.plot_met_profile(part_matrix, part_samples, X)
    pdf_map.savefig(fig)
    plt.close(fig)
    
    X,Z = model.viterbi(met_prob, lambda s,o: emission_probs(s,o,segment_p[sample_map,:],segment_prior))
    X = cleanup_segmentation(segment_p, X)
    write_region(ml_f,X)

    fig = plt.figure(figsize=(12,6))
    nanoepiseg.helper.plotting.plot_met_profile(part_matrix, part_samples, X)
    pdf_ml.savefig(fig)
    plt.close(fig)

    start+=ws
pdf_ml.close()
pdf_map.close()
ml_f.close()
map_f.close()
