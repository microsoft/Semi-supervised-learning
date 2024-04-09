import numpy as np
from sklearn.datasets import make_classification

from math import gamma
from operator import mul
import numpy as np
from functools import reduce

np.random.seed(42)

class Dirichlet(object):
    '''
    Based on http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
    '''
    def __init__(self, alpha):
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])

    def sample(self, size=None, **kwargs):
        return np.random.dirichlet(self._alpha, size=size, **kwargs)

    def __str__(self):
        return np.array2string(self._alpha, separator=',', precision=2)

    def __repr__(self):
        return self._alpha.__repr__()


class MixtureDistribution(object):
    def __init__(self, priors, distributions):
        self.priors = np.array(priors)
        self.distributions = distributions

    def sample(self, size=None):
        if size is None:
            size = len(self.priors)
        classes = np.random.multinomial(n=1, pvals=self.priors, size=size)
        samples = np.empty_like(classes, dtype='float')
        for i, size in enumerate(classes.sum(axis=0)):
            samples[np.where(classes[:,i])[0]] = self.distributions[i].sample(size)
        return samples, classes

    def posterior(self, pvalues, c=0):
        likelihoods = np.array([d.pdf(pvalues) for d in self.distributions])
        Z = np.dot(likelihoods, self.priors)
        return np.divide(likelihoods[c]*self.priors[c], Z)

    def pdf(self, pvalues):
        likelihoods = np.array([d.pdf(pvalues) for d in self.distributions])
        return np.dot(self.priors, likelihoods)

    def __repr__(self):
        string = ''
        for p, d in zip(self.priors, self.distributions):
            string += 'prior = {}, '.format(p)
            string += 'Distribution = {}'.format(d)
            string += '\n'
        return string

def get_simple_binary_example():
    S = np.random.beta(10, 0.1, 100).transpose()
    S[50:] = 1-S[50:]
    S = np.vstack([S, 1-S]).T
    y = np.hstack([np.zeros(50), np.ones(50)])
    return S, y

def get_extreme_binary_example():
    S = np.random.beta(10000, 0.1, 100).transpose()
    S[50:] = 1-S[50:]
    S = np.vstack([S, 1-S]).T
    y = np.hstack([np.zeros(50), np.ones(50)])
    return S, y

def get_simple_ternary_example():
    mdir = MixtureDistribution([1./3, 1./3, 1./3],
                               [Dirichlet([5, 0.2, .3]),
                                Dirichlet([1, 5, .5]),
                                Dirichlet([.1, 2, 10])])
    S, Y = mdir.sample(1000)
    y = Y.argmax(axis=1)
    return S, y
