import numpy as np
from scipy.stats import lognorm as scipy_lognorm, norm as scipy_norm

# development stuff
from numpy.linalg.linalg import LinAlgError
from traceback import format_exc
from pdb import set_trace
from logging import getLogger
from warnings import warn
logger = getLogger(__name__)

class RandomVariable(object):
    """ random variable object. Implement methods that scipy rvs have here """

    def __getattr__(self, name, *args, **kwargs):
        """ by default, the method in rv should be called """
        return getattr(self.rv, name, *args, **kwargs)


    def __str__(self):
        s = '%s rv. ' % self.__class__.__name__
        s += 'mean=%.2f, var=%.2f, skew=%.2f, kurt=%.2f' % self.rv.stats(moments='mvsk')
        return s


    def logpdf_grad(self, x):
        raise NotImplementedError('')


class lognorm(RandomVariable):
    """ lightnly wrapped function which allows for more intuitive initialiation """
    def __init__(self, lognorm_mean=None, lognorm_mode=1., lognorm_std=10., **kwargs):
        # get the lognorm parameters (notes Jan 9, 18)
        assert lognorm_std > 0
        if lognorm_mean is not None: # then the mean is specified
            assert lognorm_mode is None, "cannot specify both lognorm mean and mode"
            assert lognorm_mean > 0
            expsig2 = (float(lognorm_std)/float(lognorm_mean))**2 + 1.
            lognorm_sig = np.sqrt(np.log(expsig2))
            lognorm_scale = float(lognorm_mean) / np.sqrt(expsig2)
        else: # then the mode is specified
            assert lognorm_mode > 0
            expsig2 = np.roots([float(lognorm_mode)**2, -float(lognorm_mode)**2, 0., 0., -float(lognorm_std)**2])
            expsig2 = np.real(expsig2[np.logical_and(np.isreal(expsig2), expsig2 > 0)]).squeeze() # only one real positive root
            assert expsig2.size == 1, "error in the polynomial root finding for lognormal mode"
            lognorm_sig = np.sqrt(np.log(expsig2))
            lognorm_scale = float(lognorm_mode) * expsig2

        # now initialize the scipy object
        loc = 0. # default loc
        self.rv = scipy_lognorm(s=lognorm_sig, scale=lognorm_scale, loc=0., **kwargs)
        self._s = lognorm_sig
        self._scale = lognorm_scale
        self._loc = loc
        self._s2 = self._s**2


    def logpdf_grad(self, x):
        """ compute the lognorm gradient """
        # https://www.wolframalpha.com/input/?i=d(log(e%5E(-(log((x-m)%2Fc))%5E2%2F(2+s%5E2))%2F(sqrt(2*pi)+s+((x-m)%2Fc))%2Fc))%2Fdx
        return (self._s2 + np.log((x-self._loc)/self._scale))/((self._loc-x)*self._s2)


class norm(RandomVariable):
    """ lightnly wrapped function which allows for more natural initialiations for priors """
    def __init__(self, mean=0., std=10., **kwargs):
        self.rv = scipy_norm(loc=mean, scale=std, **kwargs)


class StreamMeanVar(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    Based off code from here https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream

    Inputs can be vectors to hanle multiple streams
    """

    def __init__(self, iterable=None, ddof=0):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)


    def include(self, datum):
        if self.n == 0: # if first iteration
            assert np.ndim(datum) <= 1
            self.n_streams = np.size(datum)
        else: # else if not first iteration
            assert np.shape(datum) == (self.n_streams,), "number of streams inconsistent with previous inclusions"
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)
