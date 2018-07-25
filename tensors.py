from .linalg import log_kron
import numpy as np
import scipy.linalg as la
import scipy.linalg.blas as blas
import scipy.sparse as sparse
from numpy.linalg.linalg import LinAlgError
from copy import deepcopy
from logging import getLogger
logger = getLogger(__name__)
from pdb import set_trace
from warnings import warn

class KronMatrix(object):
    """
    Tensor class which is a Kronecker product of matricies.

    Trefor Evans
    """

    def __init__(self, K, sym=False):
        """
        Inputs:
            K  : is a list of numpy arrays
            sym : bool (default False)
                specify true it submatricies in K are symmetric
        """
        self._K = K # shallow copy
        self.n = len(self.K)
        self.sshape = np.vstack([np.shape(Ki) for Ki in self.K]) # sizes of the sub matrices
        self.shape = np.atleast_1d(np.prod(np.float64(self.sshape),axis=0)) # shape of the big matrix
        if np.all(self.shape < np.iinfo(np.uint64).max): # then can be stored as an int else proceed with shape as a float
            self.shape = np.uint64(self.shape)
        self.ndim = self.shape.size
        assert self.ndim <= 2, "kron matrix cannot be more than 2d"
        self.square = self.ndim==2 and self.shape[0]==self.shape[1]
        self.sym = sym
        if sym:
            assert np.array_equal(self.sshape[:,0],self.sshape[:,1]), 'this matrix cannot be symmetric: it is not square'
            self = self.ensure_fortran()

    @property
    def K(self):
        return self._K
    @K.setter
    def K(self,K):
        raise AttributeError("Attribute is Read only.")


    def kronvec_prod(self, x):
        """
        Computes K*x where K is a kronecker product matrix and x is a column vector
        which is a numpy array

        Inputs:
            self : (N,M) KronMatrix
            x : (M,1) matrix

        Outputs:
            y : (N,1) matrix

        this routine ensures all matricies are in fortran contigous format before using blas routines

        it also takes advantage of matrix symmetry (if sym flag is set)
        """
        if x.shape != (self.shape[1],1):
            raise ValueError('x is the wrong shape, must be (%d,1), not %s' % (self.shape[1],repr(x.shape)))

        y = x
        for i,Ki in reversed(list(enumerate(self.K))):
            y = np.reshape(y, (self.sshape[i,1], -1),order='F')
            if isinstance(Ki, np.ndarray): # use optimized blas routines
                if np.isfortran(Ki):
                    a = Ki
                else:
                    a = Ki.T
                if self.sym:
                    if np.isfortran(y):
                        y = blas.dsymm(alpha=1, a=a, b=y  , side=0).T
                    else:
                        y = blas.dsymm(alpha=1, a=a, b=y.T, side=1) # just switch the order
                else:
                    if np.isfortran(y):
                        b = y
                    else:
                        b = y.T
                    y = blas.dgemm(alpha=1, a=a, b=b, trans_a=(not np.isfortran(Ki)), trans_b=(not np.isfortran(y))).T
            else: # use __mul__ routine
                y = (Ki*y).T
        y = y.reshape((-1,1),order='F') # reshape to a column vector
        return y


    def __mul__(self,x):
        """ overloaded * operator """
        return self.kronvec_prod(x)


    def kronkron_prod(self, X):
        """
        computes K*X where K,X is a kronecker product matrix
        """
        if not isinstance(X,KronMatrix):
            raise TypeError("X is not a KronMatrix")
        elif X.n != self.n:
            raise TypeError('inconsistent kron structure')
        elif not np.array_equal(X.sshape[1],self.sshape[0]):
            raise TypeError("Dimensions of X submatricies are not consistent")
        # perform the product
        return KronMatrix([self.K[i].dot(X.K[i]) for i in range(self.n)])


    def kronvec_div(self, x):
        """
        Computes y = K \ x where K is a kronecker product matrix and x is a column matrix

        Inputs:
            self : (N,N) triangular cholesky KronMatrix from chol
            x : (N,1) matrix

        Outputs:
            y : (N,1) matrix
        """
        assert self.ndim == 2
        if x.shape != (self.shape[0],1):
            raise ValueError('x is the wrong shape, must be (%d,1)' % self.shape[0])

        y = x # I don't care to actually copy this explicitly
        for i,Ki in enumerate(self.K):
            y = np.reshape(y, (-1, self.sshape[i,0]),order='F')
            if hasattr(Ki, "solve"):
                y = Ki.solve(b=y.T)
            else:
                y = la.solve(a=Ki, b=y.T, sym_pos=self.sym, overwrite_b=True)
        y = y.reshape((-1,1), order='F')
        return y


    def chol(self):
        """
        performs cholesky factorization, returning upper triangular matrix

        see equivalent in matlab for details
        """
        assert self.square
        C = np.empty(self.n, dtype=object)
        for i,Ki in enumerate(self.K):
            if hasattr(Ki, "chol"):
                C[i] = Ki.chol() # its assumed that it will return upper triangular!
            else:
                C[i] = np.linalg.cholesky(Ki).T
        return KronMatrix(C)


    def schur(self):
        """ compute schur decomposition. Outputs (Q,T). """
        assert self.square
        T = np.empty(self.n, dtype=object)
        Q = np.empty(self.n, dtype=object)
        for i,Ki in enumerate(self.K):
            if hasattr(Ki, "schur"):
                T[i],Q[i] = Ki.schur()
            else:
                T[i],Q[i] = la.schur(Ki)
        return KronMatrix(Q), KronMatrix(T)


    def svd(self):
        """
        singular value decomposition
        NOTE: it is currently assumed that self is PSD. This can easily be changed tho
        Returns (Q,eig_vals) where Q is a KronMatrix whose columns are eigenvalues and eig_vals is a 1D array
        """
        assert self.square, "for this implementation the matrix needs to be square for now"
        # first get a list of the 1D matricies
        #(Q,eig_vals) = zip(*[np.linalg.svd(Ki, full_matrices=0, compute_uv=1)[:2] for Ki in self.K])
        Q = np.empty(self.n, dtype=object)
        eig_vals = np.empty(self.n, dtype=object)
        for i,Ki in enumerate(self.K):
            try:
                if hasattr(Ki, "svd"):
                    (Q[i], eig_vals[i]) = Ki.svd()
                else:
                    (Q[i], eig_vals[i]) = np.linalg.svd(Ki, full_matrices=0, compute_uv=1)[:2]
            except LinAlgError:
                logger.error('SVD failed on dimension %d.' % (i))
                if isinstance(Ki, np.ndarray):
                    logger.error('rcond=%g.' % (np.linalg.cond(Ki)))
                raise
        return KronMatrix(Q), KronMatrix(eig_vals)


    def transpose(self):
        """ transpose the kronecker product matrix. Won't copy the matricies but will return a view """
        assert self.ndim == 2
        if self.sym:
            return self
        else:
            return KronMatrix([Ki.T for Ki in self.K]) # transpose each submatrix
    T = property(transpose) # calling self.T will do the same thing as transpose

    def expand(self, log_expansion=False):
        """
        expands the kronecker product matrix explicitly. Expensive!

        Inputs:
            log_expansion : if used then will perform a numerically stable expansion.
                If specified then the output will be the log of the value.
        """
        if log_expansion:
            Kb = np.array([0.])
            for Ki in self.K:
                if hasattr(Ki, "expand"):
                    Kb = log_kron(a=Kb, b=Ki.expand(), a_logged=True)
                else:
                    Kb = log_kron(a=Kb, b=Ki, a_logged=True)
        else:
            Kb = 1.
            if self.ndim == 1 and self.n > 10:
                warn('consider using the log_expansion options which will be more numerically stable')
            for Ki in self.K:
                if hasattr(Ki, "expand"):
                    Kb = np.kron(Kb, Ki.expand())
                else:
                    Kb = np.kron(Kb, Ki)
        return Kb.reshape(np.int32(self.shape))


    def inv(self):
        """ invert matrix """
        assert self.square
        I = np.empty(self.n, dtype=object)
        for i,Ki in enumerate(self.K):
            if hasattr(Ki, "inv"):
                I[i] = Ki.inv()
            else:
                I[i] = np.linalg.inv(Ki)
        return KronMatrix(I)


    def diag(self):
        """
        returns the diagonal of the kronecker product matrix as 1d KronMatrix
        """
        assert self.ndim == 2
        D = np.empty(self.n, dtype=object)
        for i,Ki in enumerate(self.K):
            if hasattr(Ki, "diag"):
                D[i] = Ki.diag()
            else:
                D[i] = np.diag(Ki)
        return KronMatrix(D)


    def sub_cond(self):
        """ return the condition number of each of the sub matricies """
        assert self.square
        return [np.linalg.cond(Ki) for Ki in self.K]


    def sub_shift(self,shift=1e-6):
        """ apply a diagonal correction term to the sub matrices to improve conditioning """
        if not np.array_equal(self.sshape[:,0],self.sshape[:,1]):
            raise RuntimeError('can only apply sub_shift for square matricies')
        for i,Ki in enumerate(self.K):
            self.K[i] = Ki + shift*np.identity(self.sshape[i,0])
        if self.sym:
            self = self.ensure_fortran()
        return self


    def ensure_fortran(self):
        """ ensures that the submatricies are fortran contiguous """
        for i,Ki in enumerate(self.K):
            if isinstance(Ki, np.ndarray):
                self.K[i] = np.asarray(Ki, order='F')
        return self


    def solve_chol(U,x):
        """
        solves y = U \ (U' \ x)

        Inputs:
            U : (N,N) triangular cholesky KronMatrix from chol
            x : (N,1) matrix

        Outputs:
            y : (N,1) matrix
        """
        if x.shape != (U.shape[0],1):
            raise ValueError('x is the wrong shape, must be (%d,1)' % U.shape[0])

        y = x # I don't care to actually copy this explicitly
        for i,Ui in enumerate(U.K):
            y = np.reshape(y, (-1, U.sshape[i,0]),order='F')
            if hasattr(Ui, "solve_chol"):
                y = Ui.solve_chol(y.T)
            else:
                y = la.solve_triangular(a=Ui, b=y.T, trans='T', lower=False, overwrite_b=True) # y = Ui' \ y'
                y = la.solve_triangular(a=Ui, b=y,   trans='N', lower=False, overwrite_b=True) # Ui \ y
        y = y.reshape((-1,1), order='F')
        return y


    def solve_schur(Q,t,x,shift=0.0):
        """
        solves shifted linear system of equations (K+lam*I)y=x using the schur decomposition of K

        Inputs:
            Q : (N,N) unitary KronMatrix containing the eigenvectors of K
            t : (N,) array containing corresponding eigenvalues of K.
                This can be computed from the diagonal matrix T returned by schur (such that Q*T*Q.T = K) as
                t = T.diag().expand()
                if this is a KronMatrix then it will be assumed that T was passed
            x :   (N,1) matrix
            shape : float corresponding to the shift to be applied to the system

        Outputs:
            y : (N,1) matrix
        """
        if x.shape != (Q.shape[0],1):
            raise ValueError('x is the wrong shape, must be (%d,1)' % Q.shape[0])
        if isinstance(t,KronMatrix): # then assume that the T was passed
            t = t.diag().expand()
        y = (Q.T)*x
        y = y / np.reshape(t+shift, y.shape) # solve the diagonal linear system
        y = Q*y
        return y


    def eig_vals(self):
        """ returns the eigenvalues of the matrix """
        assert self.ndim == 2
        eigs = np.empty(self.n, dtype=object)
        for i,Ki in enumerate(self.K):
            if hasattr(Ki, "eig_vals"):
                eigs[i] = Ki.eig_vals()
            elif self.sym:
                eigs[i] = np.linalg.eigvalsh(Ki)
            else:
                eigs[i] = np.linalg.eigvals(Ki)
        return KronMatrix(eigs)


    def find_extremum_eigs(eigs, n_eigs, mode='largest', log_expand=False, sort=True, compute_global_loc=False):
        """
        returns the position of the n_eigs largest eigenvalues in the KronMatrix vector eigs

        Inputs:
            n_eigs : number of eigenvalues to find
            mode : largest or smallest eigenvalues
            log_expand : if true then will return the log of the eigenvalues.
                Note that this is more numerically stable if the problem is high dimensional
            sort : if true then will sort the returned eigenvalues in descenting order

        Notes:
        This function only requires at most O( (d-1)pm ) time where d is the number of dimensions,
        p is n_eigs and m is the size of each submatrix in eigs (such that eigs has size m^d).
        """
        assert eigs.ndim == 1, "eigs must be a 1D KronMatrix"
        assert isinstance(n_eigs, (int,long, np.int32)), "n_eigs=%s must be an integer" % repr(n_eigs)
        assert n_eigs >= 1, "must use at least 1 eigenvalue"
        assert n_eigs <= eigs.shape[0], "n_eigs is greater then the total number of eigenvalues"
        assert mode == 'largest' or mode == 'smallest'
        if not log_expand and eigs.n > 10:
            warn('should use log option which will be more numerically stable')
        # TODO: if all eigenvalues are requested then can save a lot of work, consider this special case

        # define a function which returns the n_eigs extremum values and indices of those values
        def get_extremum(vec):
            if np.size(vec) <= n_eigs: # then return all
                return np.arange(n_eigs), vec
            # now make the partition. this partition runs in linear O(n) time
            if mode == 'largest':
                ind = np.argpartition(vec, -n_eigs)[-n_eigs:]
            elif mode == 'smallest':
                ind = np.argpartition(vec, n_eigs)[:n_eigs]
            return ind, vec[ind]

        # break the problem up into a sequence of 2d kronecker products
        eig_loc, eig_vals = get_extremum(eigs.K[0]) # first initialize
        eig_loc = eig_loc.reshape((-1,1))
        if log_expand: # then take the log of these values
            eig_vals = np.log(eig_vals)
        for i in range(1, eigs.n):
            # perform the kronecker product and get the n_eigs extreme values
            if log_expand:
                inds, eig_vals = get_extremum(log_kron(a=eig_vals, b=eigs.K[i], a_logged=True))
            else:
                inds, eig_vals = get_extremum(np.kron(eig_vals, eigs.K[i]))
            # now update eig_loc
            eig_loc = np.hstack(
                [
                eig_loc[np.int32(np.floor_divide(inds, eigs.K[i].size)),:].reshape((inds.size,-1)), # this is the position from the previous eig_vals
                        np.int32(np.mod(         inds, eigs.K[i].size)).reshape((-1,1)) # this is the position from the current dimension eig_vals
                ])

        # now compute the global location of the eigenvalue (if your were to expand eigs, the index of each)
        if compute_global_loc:
            global_loc = np.zeros(n_eigs, dtype=int)
            cum_size = 1 # initialize the size of the vector being constructed from the previous loop
            for i in reversed(range(eigs.n)): # loop backwards
                global_loc = cum_size * eig_loc[:,i] + global_loc
                cum_size *= eigs.K[i].size # increment the vector size
        else:
            global_loc = None

        # now sort if ness
        if sort:
            order = np.argsort(eig_vals)[::-1] # descending order
            eig_vals = eig_vals[order]
            eig_loc = eig_loc[order]
            if compute_global_loc:
                global_loc = global_loc[order]
        return eig_loc, eig_vals, global_loc


    def get_col(self, pos):
        """
        returns the expanded column as a KronMatrix

        pos should be a tuple of length self.n whose elements specify the column of the sub-matrix involved in the expansion

        to return the row, do K.T.get_col(...)
        """
        assert len(pos) == self.n
        assert np.size(pos[0]) == 1
        assert isinstance(pos[0], int)
        assert self.ndim == 2
        return KronMatrix([self.K[i][:,j].reshape((-1,1)) for i,j in enumerate(pos)])


    def log_det(eig_vals):
        """ compute the log determinant in an efficient manner """
        assert eig_vals.ndim == 1 # must be 1d KronMatrix
        ldet = 0
        for i,eigs in enumerate(eig_vals.K):
            repetition = np.prod(np.delete(eig_vals.sshape,i)) # number of times this sum term is used in the log det expansion
            ldet += repetition * np.sum(np.log(eigs))
        return ldet


class SelectionMatrix:
    """ allows efficient multiplication with a selection matrix and its transpose """
    ndim = 2

    def __init__(self, indicies):
        """
        creates a selection matrix with one nonzero entry per row

        Inputs:
            indicies : bool array or tuple
                specifies the location of the non-zero in each row.
                if bool:
                    Each the index of each True element will be on its own row
                if tuple:
                    must be (selection_inds, size) where selection inds is a 1d int array and size is an int
        """
        if isinstance(indicies, tuple):
            assert len(indicies) == 2
            assert indicies[0].ndim == 1
            self.shape = [indicies[0].size, indicies[1]]
            int_idx = indicies[0]
        else:
            assert indicies.ndim == 1
            assert indicies.dtype == bool
            self.shape = [np.count_nonzero(indicies), indicies.size]
            int_idx = np.nonzero(indicies)[0]

        nnz = self.shape[0]
        self.sel = sparse.csr_matrix((np.ones(nnz,dtype=bool),(np.arange(nnz),int_idx)), shape=self.shape, dtype=bool)
        self.sel_T = self.sel.T # testing has shown the precomputing the transpose saves lots of time
        return


    def mul(self,x):
        """ matrix-vector product """
        return self.sel * x


    def mul_T(self,x):
        """ matrix-vector product with the transpose """
        return self.sel_T * x


class SelectionMatrixSparse:
    """
    allows efficient multiplication with a selection matrix and its transpose where we
    never want to explictly form a vector of full size because it is too large
    """
    ndim = 2

    def __init__(self, indicies):
        """
        creates a selection matrix with one nonzero entry per row

        Inputs:
            indicies : bool array or tuple
                specifies the location of the non-zero in each row.
                must be (selection_inds, size) where selection inds is a 1d int array and size is an int
        """
        assert isinstance(indicies, tuple)
        assert len(indicies) == 2
        assert indicies[0].ndim == 1
        self.shape = [indicies[0].size, indicies[1]]
        self.indicies = indicies[0]
        self.unique, self.unique_inverse = np.unique(self.indicies, return_inverse=True) # these are for doing matvecs with unique


    def mul(self,x):
        """ matrix product """
        assert x.ndim == 2
        return x[self.indicies,:]
    dot = __mul__ = mul # make this do the same thing


    def mul_unique(self, x):
        """
        matrix product with the unique sliced elements of x
        after mul_unique has been called, to recover the full, non-unique entires then `full = unique[S.unique_inverse]`
        """
        assert x.ndim == 2
        return x[self.unique,:]


    def mul_T(self,x):
        """ matrix-vector product with the transpose """
        raise NotImplementedError('still need to finish this. ')
        #assert x.ndim == 2
        #assert x.shape[0] == self.shape[0]
        #if self.sparse: # then return sparse matrix
            #if x.shape[1] != 1:
                #raise NotImplementedError('need to extend this to higher number of cols of x')
            #if self.indicies.dtype == bool:
                #y = sparse.csc_matrix((x.squeeze(),(self.indicies.nonzero()[0],np.zeros(x.shape[0],dtype=bool))),
                               #shape=(self.shape[0],x.shape[1]))
            #else: # it is an integer array
                #y = sparse.csc_matrix((x.squeeze(),(self.indicies,np.zeros(x.shape[0],dtype=bool))),
                               #shape=(self.shape[0],x.shape[1]))
        #else:
            #y = self._scratch
            ## I don't need to set this to zeros since self.indicies is the same every time
            #y[self.indicies,:] = x
        return y


    def __getitem__(self,key):
        if isinstance(key,tuple): # only care about first index
            key = key[0]
        return SelectionMatrixSparse(indicies=(np.atleast_1d(self.indicies[key]),self.shape[1]))


class BlockMatrix(object):
    """ create Block matrix """

    def __init__(self, A):
        """
        Builds a block matrix with which matrix-vector multiplication can be made.

        Inputs:
            A : numpy object array of blocks of size (h, w)
                i.e. A = np.array([[ A_11, A_12, ... ],
                                   [ A_21, A_22, ... ], ... ]
                Each block in A must have the methods
                * shape
                * __mul__
                * T (transpose property)
                * expand (only nessessary if is to be used)
        """
        assert A.ndim == 2, 'A must be 2d'
        self.A = A # shallow copy

        # get the shapes of the matricies
        self.block_shape = self.A.shape # shape of the block matrix
        self._partition_shape = ([A_i0.shape[0] for A_i0 in self.A[:,0]], [A_0i.shape[1] for A_0i in self.A[0,:]]) # shape of each partition
        self.shape = tuple([np.sum(self._partition_shape[i]) for i in range(2)]) # overall shape of the expanded matrix

        # ensure the shapes are consistent for all partitions
        for i in range(self.block_shape[0]):
            for j in range(self.block_shape[1]):
                assert np.all(A[i,j].shape == self.partition_shape(i,j)), "A[%d,%d].shape should be %s, not %s" % (i,j,repr(self.partition_shape(i,j)),repr(A[i,j].shape))

        # define how a vector passed to it should be split when a matrix vector product is taken
        self.vec_split =  np.cumsum([0,] + self._partition_shape[1], dtype='i')


    def partition_shape(self, i, j):
        """ returns the shape of A[i,j] """
        return (self._partition_shape[0][i],self._partition_shape[1][j])


    def __mul__( self, x ):
        """ matrix vector multiplication """
        assert x.shape == (self.shape[1], 1)

        # first split the vector x so I don't have to make so many slices (which is slow)
        xs = [x[self.vec_split[j]:self.vec_split[j+1],:] for j in range(self.block_shape[1])]

        # loop through each block row and perform the matrix-vector product
        y = np.empty(self.block_shape[0], dtype=object)
        for i in range(self.block_shape[0]):
            y[i] = 0 # initialize
            for j in range(self.block_shape[1]): # loop accross the row
                y[i] += self.A[i,j] * xs[j]

        # concatenate results
        y = np.concatenate(y,axis=0)
        return y


    def transpose(self):
        """ transpose the kronecker product matrix. This currently copies the matricies explicitly """
        A = self.A.copy()

        # first transpose each block individually
        for i in range(self.block_shape[0]):
            for j in range(self.block_shape[1]):
                A[i,j] = A[i,j].T

        # then, transpose globally
        A = A.T

        # then return a new instance of the object
        return self.__class__(A=A)
    T = property(transpose) # calling self.T will do the same thing as transpose

    def expand(self):
        """ expands each block matrix to form a big, full matrix """
        Abig = np.zeros(np.asarray(self.shape, dtype='i'))
        row_split = np.cumsum([0,] + self._partition_shape[0], dtype='i')
        col_split = np.cumsum([0,] + self._partition_shape[1], dtype='i')
        for i in range(int(round(self.block_shape[0]))):
            for j in range(int(round(self.block_shape[1]))):
                Abig[row_split[i]:row_split[i+1], col_split[j]:col_split[j+1]] = self.A[i,j].expand()
        return Abig


class KhatriRaoMatrix(BlockMatrix):
    """ a Khatri-Rao Matrix (block Kronecker Product matrix) """

    def __init__(self, A, partition=None):
        """
        Khatri-Rao Block Matrix.

        Inputs:
            A : list of sub matricies or 2d array of KronMatricies. If the latter then partition is ignored.
            partition : int specifying the direction that the Khatri-Rao Matrix is partitioned:
                0 : row partitioned
                1 : column partitioned
                Note that if A is an array of KronMatricies then this has now effect.
        """
        # determine whether KronMatricies have already been formed from the partitions or not
        if np.ndim(A)==2 and isinstance(A[0,0], KronMatrix): # then all the work is done
            super(KhatriRaoMatrix, self).__init__(A)
            return

        # else I need to create KronMatrices from each partition
        # get the number of blocks that will be needed
        assert partition in range(2)
        if partition == 0:
            block_shape = (A[0].shape[0], 1)
        elif partition == 1:
            block_shape = (1,A[0].shape[1])
        else:
            raise ValueError('unknown partition')

        # form the KronMatricies
        Akron = np.empty(max(block_shape), dtype=object) # make 1d now and reshape later
        for i in range(max(block_shape)):
            if partition == 0:
                Akron[i] = KronMatrix([Aj[(i,),:] for Aj in A], sym=False)
            elif partition == 1:
                Akron[i] = KronMatrix([Aj[:,(i,)] for Aj in A], sym=False)
        Akron = Akron.reshape(block_shape)

        # Create a BlockMatrix from this
        super(KhatriRaoMatrix, self).__init__(Akron)


class RowColKhatriRaoMatrix(object):
    """ matrix formed by R K C allowing memory efficient matrix-vector products """

    def __init__(self,R,K,C,nGb=1.):
        """
        Inputs:
            R : list or np.ndarray
            K : list or np.ndarray, K can be None if there is no K
            C : list or np.ndarray
            nGb : number of gigabytes of memory which shouldn't be exceeded when performing mvproducts
                If specified the multiple rows will be computed at once which allows for
                BLAS level-3 routines to be used
        Note:
            * by default, KC will be merged therefore R should be sparse if any. If C is sparse then you should either:
                use the ...Transposed class below
        """
        self.shape = (R[0].shape[0],C[0].shape[1])
        self.d = len(R)
        if K is not None:
            K = np.asarray(K)
            assert len(K) == len(C) == self.d, "number of dimensions inconsistent"

            # merge K and C into self.C, a row-partitioned Khatri-Rao product matrix
            self.R = R
            self.C = np.empty(self.d,dtype=object)
            for i in range(self.d): # ensure submatricies are consistent
                assert K[i].shape[0] == K[i].shape[1] == R[i].shape[1], "K must be a square Kronecker product matrix, and must be consistent with R"
                self.C[i] = K[i].dot(C[i])
        else: # there is no K
            self.R = R
            self.C = C

        # figure out how many rows should be computed at once during matvec
        self.n_rows_at_once = 1 # default one at a time
        if nGb is not None:
            self.n_rows_at_once = max(1,np.int32(np.floor(nGb*1e9/(8*self.shape[1]))))
        self.nGb = nGb

    @property
    def T(self):
        """ transpose operation """
        if isinstance(self.R[0], (SelectionMatrix,SelectionMatrixSparse)): # then I don't actually want to transpose explicitly
            return RowColKhatriRaoMatrixTransposed(R=self.R, K=None, C=self.C, nGb=self.nGb)
        else: # actually compute the transpose
            return RowColKhatriRaoMatrix(R=[Ci.T for Ci in self.C], K=None, C=[Ri.T for Ri in self.R], nGb=self.nGb)


    def get_rows(self, i_rows, logged=False):
        """
        compute i_rows rows of the packed matrix
        a can be a vector or a slice object (ie. i_rows=slice(None) will return the whole matrix)

        if logged then returns the log of the rows which is more numerically stable
        """
        if logged:
            rows = 0.
            sign = 1.
        else:
            rows = 1.
        for i_d in range(self.d):
            if sparse.issparse(self.C[i_d]): # have to treat this differently as of numpy 1.7
                # ... see http://stackoverflow.com/questions/31040188/dot-product-between-1d-numpy-array-and-scipy-sparse-matrix
                rows_1d = self.R[i_d][i_rows,:]  *  self.C[i_d]
            else:
                rows_1d = self.R[i_d][i_rows,:].dot(self.C[i_d])
            if logged:
                sign *= np.int32(np.sign(rows_1d))
                rows_1d[sign == 0] = 1. # if there's a zero then it doesn't matter what this value is
                rows += np.log(np.abs(rows_1d))
            else:
                rows *= rows_1d
        if logged:
            return rows, sign
        else:
            return rows


    def expand(self, logged=False):
        """
        expand the product of matricies.
        this is very similar to the get_rows routine since that just computes the matrix a few rows at a time whereas
        here we do it all in one shot since we assume it is practical to store the entire matrix in memory

        if logged then returns the log of the matrix which is more numerically stable
        """
        return self.get_rows(i_rows=slice(None), logged=logged)


    def __mul__(self,x):
        """
        memory efficient way to compute a matrix-vector product with a row and column partitioned Khatri-Rao product matrix.
        This is the same as mvKRrowcol from the ICML submission
        """
        assert x.shape == (self.shape[1],1)
        i = 0 # initialize counter
        y = np.zeros((self.shape[0],1))
        while i < self.shape[0]: # loop accross rows of the matrix
            i_rows = np.arange(i,min(i+self.n_rows_at_once,self.shape[0]))
            y[i_rows,:] = self.get_rows(i_rows).dot(x) # compute matvec
            i = i_rows[-1] + 1 # increment counter
        return y


class RowColKhatriRaoMatrixTransposed(RowColKhatriRaoMatrix):
    """ thin wrapper for when the row KR is sparse before transposing """

    def __init__(self,*args,**kwargs):
        super(RowColKhatriRaoMatrixTransposed, self).__init__(*args,**kwargs)
        self.shape = self.shape[::-1] # flip this

        # need to redo this since now really slicing columns (and then transposing)
        self.n_rows_at_once = 1 # default one at a time
        if self.nGb is not None:
            self.n_rows_at_once = max(1,np.int32(np.floor(self.nGb*1e9/(8*self.shape[1]))))


    def get_rows(self,i_rows):
        """
        compute i_rows columns of the packed matrix and then transpose
        """
        if sparse.issparse(self.C[0]): # have to treat this differently as of numpy 1.7
            # ... see http://stackoverflow.com/questions/31040188/dot-product-between-1d-numpy-array-and-scipy-sparse-matrix
            cols = self.R[0] * self.C[0][:,i_rows]
        else:
            cols = self.R[0].dot(self.C[0][:,i_rows])
        for i_d in range(1,self.d):
            if sparse.issparse(self.C[i_d]): # have to treat this differently as of numpy 1.7
                cols *= self.R[i_d]  *  self.C[i_d][:,i_rows]
            else:
                cols *= self.R[i_d].dot(self.C[i_d][:,i_rows])
        return cols.T

    @property
    def T(self):
        """ transpose operation """
        # since its already a transposed type matrix, I just need to respecify as the non-transposed class
        return RowColKhatriRaoMatrix(R=self.R, K=None, C=self.C, nGb=self.nGb)


class TensorProduct(object):
    """ class for performing matrix-vector product with a product of several tensors without expansion """

    def __init__(self, tensor_list):
        """
        all tensors must be 2d and have attribues:
            * shape

        and must have methods:
            * __mul__ (for vectors)
            * T (transpose)
        """
        self.tensors = tensor_list
        self.n_tensors = len(tensor_list)
        self.shape = (self.tensors[0].shape[0], self.tensors[-1].shape[1])

        # check to ensure the shapes are compatable
        for i in range(self.n_tensors-1):
            assert self.tensors[i].shape[1] == self.tensors[i+1].shape[0], "shapes of tensors are incompatable"

    @property
    def T(self):
        raise NotImplementedError('easy to do this')


    def __mul__(self,x):
        assert x.shape == (self.shape[1], 1), "vector is wrong shape"
        y = x
        for i in reversed(range(self.n_tensors)):
            y = self.tensors[i] * y
        return y


class TensorSum(object):
    """ class for performing matrix-vector product with a sum of several tensors without expansion """

    def __init__(self, tensor_list):
        """
        all tensors must be 2d and have attribues:
            * shape

        and must have methods:
            * __mul__ (for vectors)
            * T (transpose)
        """
        self.tensors = tensor_list
        self.n_tensors = len(tensor_list)
        self.shape = self.tensors[0].shape

        # check to ensure the shapes are compatable
        for i in range(self.n_tensors-1):
            assert self.tensors[i].shape == self.tensors[i+1].shape, "shapes of tensors are incompatable"

    @property
    def T(self):
        raise NotImplementedError('easy to do this')


    def __mul__(self,x):
        assert x.shape == (self.shape[1], 1), "vector is wrong shape"
        y = np.zeros((self.shape[0],1))
        for tensor in self.tensors:
            y += tensor * x
        return y


class Array:
    """ simple lightweight wrapper for numpy.ndarray (or others) that will work with tensor objects """
    def __init__(self,A):
        self.A = A
        self.shape = A.shape

    def __mul__(self,x):
        return self.A.dot(x)

    @property
    def T(self):
        return Array(self.A.T)

    def expand(self):
        return self.A


def expand_SKC(S, K, C, logged=True):
    """
    Expand selection matrix * kron matrix * column-partitioned KR matrix

    Inputs:
        S : list, row KR matrix of selection matricies
        K : list, kron matix
        C : list, column partitioned Khatri-Rao matrix
        logged : if logged then returns the log of the rows which is more numerically stable
    """
    assert isinstance(S, (list,np.ndarray))
    assert isinstance(S[0], SelectionMatrixSparse)
    assert isinstance(K, (list,np.ndarray))
    assert isinstance(C, (list,np.ndarray))
    if logged:
        log_prod = 0.
        sign = 1.
    else:
        prod = 1.
    for s,k,c in zip(S, K, C):
        x_unique = s.mul_unique(k).dot(c) # just compute the unique rows of the product
        if logged:
            sign *= np.int32(np.sign(x_unique))[s.unique_inverse]
            x_unique[x_unique == 0] = 1. # if there's a zero then it doesn't matter what this value is
            log_prod += np.log(np.abs(x_unique))[s.unique_inverse]
        else:
            prod *= x_unique[s.unique_inverse]
    if logged:
        return log_prod, sign
    else:
        return prod



