import time
from multiprocessing import Pool, Array
import unittest
import random
import numpy as np

from nanoepiseg.hmm import SegmentationHMM
from nanoepiseg.emissions import BernoulliPosterior
import matplotlib.pyplot as plt
import nanoepitools.plotting.plot_methylation_profile as nplt
import nanoepitools.math as nmath


def worker(obs, samples):
    n_segments = 20
    emission_lik = BernoulliPosterior(obs.shape[0], n_segments, prior_a=0.9)
    hmm = SegmentationHMM(max_segments=n_segments, t_stay=0.1, t_move=0.8, e_fn=emission_lik, eps=np.exp(-512))
    segment_p, posterior = hmm.baum_welch(obs, tol=np.exp(-8), samples=samples)
    segmentation, _ = hmm.MAP(posterior)
    return segment_p, segmentation


class HMMTestCase(unittest.TestCase):
    def test_instanciate(self):
        emission_lik = BernoulliPosterior(5, 6)
        hmm = SegmentationHMM(max_segments=10, t_stay=0.5, t_move=0.5, e_fn=emission_lik)
        self.assertIsNotNone(hmm)
    
    def test_instanciate_optional_parameters(self):
        emission_lik = BernoulliPosterior(5, 6)
        hmm = SegmentationHMM(max_segments=10, t_stay=0.5, t_move=0.2, seg_penalty=0.1, eps=0.0001, e_fn=emission_lik)
        self.assertIsNotNone(hmm)
    
    def random_data(self, n_reads, n_obs):
        # Simple up-down signal
        obs = np.repeat([[((i // 10) % 2) == 0 for i in range(n_obs)]], n_reads, axis=0).astype(float)
        obs = obs.clip(0.01, 0.99)
        
        # Make some values "missing"
        for i in range(n_reads):
            start = random.randrange(0, n_obs - 10)
            end = random.randrange(start + 10, n_obs)
            if i % 2 == 0:
                obs[i, :] = 1 - obs[i, :]
            obs[i, :start] = -1
            obs[i, end:] = -1
        obs = obs[:, (obs != -1).sum(axis=0) > 0]
        
        return obs
    
    def test_segment_simple_example(self):
        random.seed(42)
        n_obs = 100
        n_reads = 40
        n_segments = 10
        emission_lik = BernoulliPosterior(40, n_segments, prior_a=0.9)
        hmm = SegmentationHMM(max_segments=n_segments, t_stay=0.1, t_move=0.8, e_fn=emission_lik, eps=np.exp(-512))
        
        obs = self.random_data(n_reads, n_obs)
        
        start = time.time()
        segment_p, posterior = hmm.baum_welch(obs, tol=np.exp(-8), samples=np.arange(0, n_reads))
        
        segmentation, _ = hmm.MAP(posterior)
        end = time.time()
        print("Took: ", (end - start))
        
        self.assertIsNotNone(segmentation)
    
    def test_mp(self):
        random.seed(42)
        n_obs = 250
        n_reads = 100
        
        windows = [(self.random_data(n_reads, n_obs), np.arange(0, n_reads)) for _ in range(10)]
        
        p = Pool(1)
        start = time.time()
        results = p.starmap(worker, windows)
        for r in results:
            print(results[0][0][0].shape)
        end = time.time()
        print("Took: ", (end - start))


if __name__ == "__main__":
    unittest.main()
