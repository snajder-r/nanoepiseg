import os
import sys
import numpy as np


logdir='/home/r933r/snajder/nanopore/data/medulloblastoma_dna/logs/'
python='/home/r933r/.conda/envs/gastrulation/bin/python'
script='/home/r933r/code/segmentation_hmm/experiments/run_for_chr_q.py'
#for chrom  in [str(x) for x in range(2,23)]:
for chrom  in [18]:
    for start in np.arange(0,1, 0.01):
        end = min(start+0.01,1)
        command = '{python} {script} chr{chrom} {start} {end} q{start}'.format(chrom=chrom,python=python, script=script, start='%0.2f'%start,end='%0.2f'%end)
        print(command)
        bsub = 'bsub  -R "rusage[mem=8000]" -W 3:00 -o {log}.log -e {log}.err "{command}"'.format(log=logdir+'segment_%0.2f'%start, command=command)
        os.system(bsub)
