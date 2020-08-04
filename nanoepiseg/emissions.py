import numpy as np
from abc import ABC, abstractmethod
import math


class EmissionLikelihoodFunction(ABC):

    @abstractmethod
    def update_params(self, **argv):
        pass

    @abstractmethod
    def likelihood(self, segment_index: int, observartions: np.array):
        pass


class BernoulliPosterior(EmissionLikelihoodFunction):
    """
    This class models a Bernoulli likelihood with uncertain observations.
    Observations are provided as probabilities p(a|S), and the likelihood
    is parameterized with bernoulli-likelihoods p(a|mu) for each segment.
    Optionally, a gamma prior p(mu) can be defined as well.
    The likelihood is modeled as:
    L = (1-p(a|S)) * (1-mu) / (1-p(a)) + p(a|S) * mu / p(a)
    """

    segment_p: np.array = None
    segment_prior: np.array = None
    prior_lognormfactor: float

    def __init__(self, prior_a: float = None, eps=np.exp(-512)):
        self.eps = eps
        self.prior_a = prior_a

        if self.prior_a is not None:
            # Precompute the normfactor (in log space) of the prior gamma
            # distribution
            self.prior_lognormfactor = np.log(math.gamma(2 * prior_a) / (
                    math.gamma(prior_a) ** 2))

    def update_prior(self):
        """
        Updates the prior beta distribution (in log space) based on
        self.segment_p and self.prior_a
        """
        self.segment_prior = self.segment_p * (self.prior_a - 1)
        self.segment_prior += np.log(1 - np.exp(self.segment_p) + self.eps) *\
                              (self.prior_a - 1)
        self.segment_prior += self.prior_lognormfactor

    def update_params(self, segment_p: np.array):
        self.segment_p = segment_p
        if self.prior_a is not None:
            self.update_prior()

    def likelihood(self, segment_index: int, observations: np.array):
        idx = (observations != -1)
        ret_a = (np.log(1 - np.exp(self.segment_p[idx, segment_index]) +
                        self.eps)) + np.log(1 - observations[idx]) + np.log(0.5)

        ret_b = self.segment_p[idx, segment_index]
        ret_b += np.log(observations[idx]) + np.log(0.5)

        ret = np.logaddexp(ret_a, ret_b)
        if self.segment_prior is not None:
            ret += self.segment_prior[idx, segment_index]
        return ret.sum()

    def minimization_objective(self, observations, posterior):
        def curried_objective(x):
            M = self.segment_p.shape[1]
            R = observations.shape[0]
            ls = np.zeros(M)
            ps = np.zeros(M)
            for r in range(R):
                o = observations[r, :]
                idx = o != -1
                o = o[idx]
                pki = posterior[idx, :]

                l_a = np.outer((1 - o), (1 - x)) / 2
                l_b = np.outer(o, x) / 2
                l = np.log(l_a + l_b + self.eps)
                ls += (l * np.exp(pki)).sum(axis=0)
                ps += np.exp(pki).sum(axis=0)

            ret = ls / ps
            ret = ret.sum()
            return -ret
        return curried_objective

