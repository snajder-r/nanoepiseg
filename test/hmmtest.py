import unittest
import random
import numpy as np

from nanoepiseg.hmm import SegmentationHMM
from nanoepiseg.emissions import BernoulliPosterior
import matplotlib.pyplot as plt
import nanoepitools.plotting.plot_methylation_profile as nplt
import nanoepitools.math as nmath


class HMMTestCase(unittest.TestCase):

    def test_instanciate(self):
        emission_lik = BernoulliPosterior()
        hmm = SegmentationHMM(max_segments=10, t_stay=0.5, t_move=0.5,
                              e_fn=emission_lik)
        self.assertIsNotNone(hmm)

    def test_instanciate_optional_parameters(self):
        emission_lik = BernoulliPosterior()
        hmm = SegmentationHMM(max_segments=10, t_stay=0.5, t_move=0.2,
                              seg_penalty=0.1, eps=0.0001, e_fn=emission_lik)
        self.assertIsNotNone(hmm)

    def test_segment_simple_example(self):
        random.seed(42)
        emission_lik = BernoulliPosterior(prior_a=0.9)
        hmm = SegmentationHMM(max_segments=30, t_stay=0.1, t_move=0.85,
                              e_fn=emission_lik, eps=np.exp(-512))
        n_obs = 100
        n_reads = 40
        # Simple up-down signal
        obs = np.repeat([[((i // 10) % 2) == 0 for i in range(n_obs)]],
                        n_reads, axis=0).astype(float)
        obs = obs.clip(0.01, 0.99)

        # Make some values "missing"
        for i in range(n_reads):
            start = random.randrange(0, n_obs-10)
            end = random.randrange(start+10, n_obs)
            obs[i, start:end] = -1

        segment_p, posterior = hmm.baum_welch(obs, tol=np.exp(-8))

        segmentation, _ = hmm.MAP(posterior)
        print(segmentation)

        obs[obs != -1] = nmath.p_to_llr(obs[obs != -1])
        obs[obs == -1] = 0
        nplt.plot_met_profile(obs, samples=np.repeat('1', n_reads),
                              sample_order=['1'], sample_colors={'1': 'r'},
                              segment=segmentation)
        plt.savefig('/home/r933r/test.png')

        self.assertIsNotNone(posterior)


if __name__ == '__main__':
    unittest.main()