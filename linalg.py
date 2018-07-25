from scipy.linalg import solve_triangular
import numpy as np
import logging
import warnings
logger = logging.getLogger(__name__)
from pdb import set_trace
import sys


def solve_schur(Q,t,x,shift=0.0):
    """
    solves shifted linear system of equations (K+shift*I)y=x using the schur decomposition of K

    Inputs:
        Q : (N,N) unitary KronMatrix containing the eigenvectors of K
        t : (N,) array containing corresponding eigenvalues of K.
            This can be computed from the diagonal matrix T returned by schur (such that Q*T*Q.T = K) as
            t = T.diag()
        x :   (N,1) matrix
        shift : float corresponding to the shift to be applied to the system

    Outputs:
        y : (N,1) matrix
    """

    if x.shape != (Q.shape[0],1):
        raise ValueError('x is the wrong shape, must be (%d,1)' % Q.shape[0])
    y = np.dot(Q.T, x)
    y = y / np.reshape(t+shift, y.shape) # solve the diagonal linear system
    y = np.dot(Q, y)
    return y


def solve_chol(U,x):
    """
    solves y = U \ (U' \ x)

    Inputs:
        U : (N,N) upper triangular cholesky cholesky factorized matrix
        x : (N,1) matrix

    Outputs:
        y : (N,1) matrix
    """
    if x.shape != (U.shape[0],1):
        raise ValueError('x is the wrong shape, must be (%d,1)' % U.shape[0])
    y = solve_triangular(U, x, trans=1, lower=False, check_finite=False) # y = Ui' \ x
    y = solve_triangular(U, y, trans=0, lower=False, check_finite=False) # Ui \ y
    return y


class solver_counter:
    """
    counter for pcg, gmres, ... scipy routines since they don't keep count of iterations. see here:
        http://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method
    """

    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.backup= None


    def __call__(self, rk=None, msg='', store=None):
        self.niter += 1
        if self._disp:
            logger.info('iter %3i. %s' % (self.niter,msg))
            sys.stdout.flush()
        if store is not None: # then backup the value
            self.backup = store


def log_kron(a, b, a_logged=False, b_logged=False):
    """
    computes np.log(np.kron(a,b)) in a numerically stable fashion which
    decreases floating point error issues especially when many kron products
    are evaluated sequentially

    Inputs:
        a_logged : specify True if the logarithm of a has already been taken
        b_logged : specify True if the logarithm of b has already been taken
    """
    assert a.ndim == b.ndim == 1, "currenly only working for 1d arrays"
    if not a_logged:
        a = np.log(a)
    if not b_logged:
        b = np.log(b)
    return (a.reshape((-1,1)) + b.reshape((1,-1))).reshape(-1)


def uniquetol(x, tol=1e-6, relative=False):
    """
    return unique values in array to within a tolerance

    Inputs:
        x : 1d array
        tol : threshold tolerance. relative changes this behavour
        relative : if relative is true then the tolerance is scaled by the range of the data
    """
    assert x.ndim == 1
    if relative: # then scale tol
        tol = np.float64(tol) * np.ptp(x)
    return x[~(np.triu(np.abs(x[:,None] - x) <= tol,1)).any(0)]


class LogexpTransformation:
    """ apply log transformation to positive parameters for optimization """
    _lim_val = 36.
    _log_lim_val = np.log(np.finfo(np.float64).max)


    def inverse_transform(self, x):
        return np.where(x > self._lim_val, x, np.log1p(np.exp(np.clip(x, -self._log_lim_val, self._lim_val))))


    def transform(self, f):
        return np.where(f > self._lim_val, f, np.log(np.expm1(f)))


    def transform_grad(self, f, grad_f):
        return grad_f*np.where(f > self._lim_val, 1.,  - np.expm1(-f))


