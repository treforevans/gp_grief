import numpy as np
from numpy import pi
from itertools import product
from .tensors import KronMatrix, KhatriRaoMatrix, BlockMatrix, SelectionMatrixSparse, RowColKhatriRaoMatrix, expand_SKC
from .grid import InducingGrid
import logging
import GPy.kern
logger = logging.getLogger(__name__)
from pdb import set_trace


class BaseKernel(object):
    """ base class for all kernel functions """

    def __init__(self, n_dims, active_dims, name):
        self.n_dims = n_dims
        if active_dims is None: # then all dims are active
            active_dims = np.arange(self.n_dims)
        else: # active_dims has been specified
            active_dims = np.ravel(active_dims) # ensure 1d array
            assert 'int' in active_dims.dtype.type.__name__ # ensure it is an int array
            assert active_dims.min() >= 0 # less than zero is not a valid index
            assert active_dims.max() <  self.n_dims # max it can be is n_dims-1
        self.active_dims = active_dims
        if name is None: # then set a default name
            name = self.__class__.__name__
        self.name = name

        # initialize a few things that need to be implemented for new kernels
        self.parameter_list = None # list of parameter attribute names as strings
        self.constraint_map = None # dict with elements in parameter_list as keys
        self._children = [] # contains the children kernels (form __mul__ and __add__)


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        x,z = self._process_cov_inputs(x,z) # process inputs
        raise NotImplementedError('Not implemented')

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        # check if implemented
        if self.parameter_list is None:
            raise NotImplementedError('Need to specify kern.parameter_list')

        # first get the parent's parameters
        parameters = [np.ravel(getattr(self, name)) for name in self.parameter_list]

        # now add the children's parameters
        parameters += [child.parameters for _,child in self._children]

        # now concatenate into an array
        if len(parameters) > 0:
            parameters = np.concatenate(parameters, axis=0)
        else:
            parameters = np.array([])
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d
        i0 = 0 # counter of current position in value

        # set the parent's parameters
        for name in self.parameter_list:
            old = getattr(self, name) # old value
            setattr(self, name, value[i0:i0+np.size(old)].reshape(np.shape(old))) # ensure same shape as old
            i0 += np.size(old) # increment counter

        # set the children's parameters
        for _,child in self._children:
            old = getattr(child, 'parameters') # old value
            setattr(child, 'parameters', value[i0:i0+np.size(old)].reshape(np.shape(old))) # ensure same shape as old
            i0 += np.size(old) # increment counter

    @property
    def constraints(self):
        """ returns the constraints for all parameters """
        # check if implemented
        if self.constraint_map is None:
            raise NotImplementedError('Need to specify kern.constraint_map')

        # first get the parent's parameters
        constraints = [np.ravel(self.constraint_map[name]) for name in self.parameter_list]

        # now add the children's parameters
        constraints += [child.constraints for _,child in self._children]

        # now concatenate into an array
        if len(constraints) > 0:
            constraints = np.concatenate(constraints, axis=0)
        else:
            constraints = np.array([])
        return constraints


    def is_stationary(self):
        """ check if stationary """
        if isinstance(self, GPyKernel):
            return isinstance(self.kern, GPy.kern.src.stationary.Stationary)
        else:
            return isinstance(self, Stationary)


    def _process_cov_inputs(self,x,z):
        """
        function for processing inputs to the cov function

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d). If none then will assume z=x

        Outputs:
            x,z
        """
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims
        if z is None:
            z = x
        else:
            assert z.ndim == 2
            assert z.shape[1] == self.n_dims, "should be %d dims, not %d" % (self.n_dims,z.shape[1])
        return x,z


    def _apply_children(self, K, x, z=None):
        """
        apply the children to the parents covariance matrix
        This MUST be called right at the end of the cov routine

        Inputs:
            K : parent's covariance matrix
            x,z : same as cov
        """
        for operation,child in self._children:
            if operation == 'mul':
                K = np.multiply(K, child.cov(x, z))
            elif operation == 'add':
                K = np.add(     K, child.cov(x, z))
            else:
                raise ValueError('Unknown kernel operation %s' % repr(operation))
        return K


    def __str__(self):
        """
        this is what is used when being printed
        """
        from tabulate import tabulate
        # print the parent
        s = '\n'
        s += "%s kernel\n" % self.name
        if isinstance(self, GPyKernel): # do this in a custom way
            s += str(tabulate([[param._name, param.values, constraint]
                               for (param, constraint) in zip(self.kern.flattened_parameters, self.constraint_list)],
                              headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl'))
        else: # tabulate the reuslts
            s += str(tabulate([[name, getattr(self, name), self.constraint_map[name]] for name in self.parameter_list],
                              headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl'))

        # now print the children
        for i,(operation,child) in enumerate(self._children):
            s += '\n\n%s with child:\n' % operation # don't add a new line here since the child will print one
            s += str(child)#.replace('\n', '\n' + '  ' * (i+1))
        s += '\n'
        return s


    def __mul__(k1,k2):
        """
        multiply kernel k1 with kernel k2
        returns k = k1 * k2
        note that explicit copies are formed in the process
        """
        assert isinstance(k2, BaseKernel)
        assert k2.n_dims == k1.n_dims # ensure the kernel has the same number of dimensions
        # copy the kernels
        parent = k1.copy()
        child  = k2.copy()

        # since multiplying, set the childs variance to fixed
        if np.size(child.constraint_map['variance']) > 1:
            child.constraint_map['variance'][0] = 'fixed'
        else:
            child.constraint_map['variance'] = 'fixed'

        # add the child to the parent
        parent._children.append( ('mul', child) ) # add the kernel the children
        return parent


    def __add__(k1,k2):
        """
        adds kernel k1 with kernel k2
        returns k = k1 + k2
        note that explicit copies are formed in the process
        """
        assert isinstance(k2, BaseKernel), 'k2 must be a kernel'  # ensure it is a kernel
        # copy the kernels
        parent = k1.copy()
        child  = k2.copy()

        # add the child to the parent
        parent._children.append( ('add', child) ) # add the kernel the children
        return parent


    def copy(self):
        """ return a deepcopy """
        from copy import deepcopy
        # first create a deepcopy
        self_copy = deepcopy(self)
        # then create a copy of all children 
        self_copy._children = [(deepcopy(operation),child.copy()) for operation,child in self_copy._children]
        return self_copy


class GPyKernel(BaseKernel):
    """ grab some kernels from the GPy library """

    def __init__(self, n_dims, kernel=None, name=None, **kwargs):
        """
        Use a kernel from the GPy library

        Inputs:
            n_dims : int
                number of input dimensions
            kernel : str OR GPy.kern.Kern
                name of the kernel in gpy OR GPy kernel object. If the latter then nothing else afterwards should be specified
                except name can be
        """
        if isinstance(kernel, str):
            if name is None:
                name = "GPy - " + kernel
            super(GPyKernel, self).__init__(n_dims=n_dims, active_dims=None, name=name) # Note that active_dims will be dealt with at the GPy level
            logger.debug('Initializing %s kernel.' % self.name)
            self.kern = eval("GPy.kern." + kernel)(input_dim=n_dims,**kwargs) # get the kernel
        elif isinstance(kernel, GPy.kern.Kern): # check if its a GPy object
            if name is None:
                name = "GPy - " + repr(kernel)
            super(GPyKernel, self).__init__(n_dims=n_dims, active_dims=None, name=name) # Note that active_dims will be dealt with at the GPy level
            logger.debug('Using specified %s GPy kernel.' % self.name)
            self.kern = kernel
        else:
            raise TypeError("must specify kernel as string or a GPy kernel object")

        # Constrain parameters  TODO: currently assuming all parameters are constrained positive, I should be able to take this directly from the flattened_parameters
        self.constraint_list = [['+ve',]*np.size(param.values) for param in self.kern.flattened_parameters]


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        K = self.kern.K(x,z)
        K = self._apply_children(K, x, z)
        return K

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        # first get the parent's parameters
        parameters = [np.ravel(param.values) for param in self.kern.flattened_parameters]

        # now add the children's parameters
        parameters += [child.parameters for _,child in self._children]

        # now concatenate into an array
        if len(parameters) > 0:
            parameters = np.concatenate(parameters, axis=0)
        else:
            parameters = np.array([])
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d
        i0 = 0 # counter of current position in value

        # set the parent's parameters
        for ip in range(np.size(self.kern.flattened_parameters)):
            old = self.kern.flattened_parameters[ip]
            try:
                self.kern.flattened_parameters[ip][:] = value[i0:i0+np.size(old)].reshape(np.shape(old)) # ensure same shape as old
            except:
                raise
            i0 += np.size(old) # increment counter

        # set the children's parameters
        for _,child in self._children:
            old = getattr(child, 'parameters') # old value
            setattr(child, 'parameters', value[i0:i0+np.size(old)].reshape(np.shape(old))) # ensure same shape as old
            i0 += np.size(old) # increment counter

    @property
    def constraints(self):
        """ get constraints. over ride of inherited property """
        # first get the parent's constraints
        constraints = [np.ravel(constraint) for constraint in self.constraint_list]

        # now add the children's constraints
        constraints += [child.constraints for _,child in self._children]

        # now concatenate into an array
        if len(constraints) > 0:
            constraints = np.concatenate(constraints, axis=0)
        else:
            constraints = np.array([])
        return constraints


    def fix_variance(self):
        """ apply fixed constraint to the variance """
        # look for the index of each occurance of variance
        i_var = np.where(['variance' in param._name.lower() for param in self.kern.flattened_parameters])[0]

        # check if none or multiple found
        if np.size(i_var) == 0:
            raise RuntimeError("No variance parameter found")
        elif np.size(i_var) >  1 or np.size(self.constraint_list[i_var[0]]) > 1:
            # ... this should be valid even when the kernel is eg. a sum of other kernels
            logger.info("Multiple variance parameters found in the GPy kernel, will only fix the first")

        # constrain it
        self.constraint_list[i_var[0]][0] = 'fixed'


class Stationary(BaseKernel):
    """ base class for stationary kernels """

    def distances_squared(self, x, z=None, lengthscale=None):
        """
        Evaluate the distance between points squared.

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional)

        Outputs:
            k : matrix of distances of shape shape (N, M)
        """
        x, z = self._process_cov_inputs(x, z) # process inputs

        # reshape the matricies correctly for broadcasting
        N = x.shape[0]
        M = z.shape[0]
        d = self.active_dims.size # the number of active dimensions
        x = np.asarray(x)[:,self.active_dims].reshape((N,1,d))
        z = np.asarray(z)[:,self.active_dims].reshape((1,M,d))

        # Code added to use different lengthscales for each dimension
        if lengthscale is None:
            lengthscale = np.ones(d,dtype='d')
        elif isinstance(lengthscale,float):
            lengthscale = lengthscale*np.ones(d,dtype='d')
        else:
            lengthscale = np.asarray(lengthscale).flatten()
            assert len(lengthscale) == d

        # now compute the distances
        return np.sum(np.power((x-z)/lengthscale.reshape((1,1,d)),2),
                      axis=2, keepdims=False)


    def distances(self, x, z=None, lengthscale=None):
        """
        Evaluate the distance between points along each dimension

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional)

        Outputs:
            k : matrix of distances of shape (N, M, d)
        """
        x, z = self._process_cov_inputs(x, z) # process inputs

        # reshape the matricies correctly for broadcasting
        N = x.shape[0]
        M = z.shape[0]
        d = self.active_dims.size # the number of active dimensions
        x = np.asarray(x)[:,self.active_dims].reshape((N,1,d))
        z = np.asarray(z)[:,self.active_dims].reshape((1,M,d))

        # Code added to use different lengthscales for each dimension
        if lengthscale is None:
            lengthscale = np.ones(d,dtype='d')
        elif isinstance(lengthscale,float):
            lengthscale = lengthscale*np.ones(d,dtype='d')
        else:
            lengthscale = np.asarray(lengthscale).flatten()
            assert len(lengthscale) == d

        # now compute the distances
        return (x-z)/lengthscale.reshape((1,1,d))


class RBF(Stationary):
    """squared exponential kernel with the same shape parameter in each dimension"""

    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs: (very much the same as in GPy.kern.RBF)
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(RBF, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        assert np.size(variance) == 1
        assert np.size(lengthscale) == 1
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None,lengthscale=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x
            lengthscale : a vector of length scales for each dimension

        Outputs:
            k : matrix of shape (N, M)
        """
        if self.lengthscale < 1e-6: # then make resiliant to division by zero
            K = self.variance * (self.distances_squared(x=x,z=z)==0) # the kernel becomes the delta funciton (white noise)
            logger.debug('protected RBF against zero-division since lengthscale too small (%s).' % repr(self.lengthscale))
        else: # then compute the nominal way
            if lengthscale is None:
                K = self.variance * np.exp( -0.5 * self.distances_squared(x=x,z=z) / self.lengthscale**2 )
            else:
                lengthscale = np.asarray(lengthscale).flatten()
                assert len(lengthscale) == self.active_dims.size
                K = self.variance * np.exp( -0.5 * self.distances_squared(x=x,z=z,lengthscale=lengthscale) )
        K = self._apply_children(K, x, z)
        return K


class Exponential(Stationary):
    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(Exponential, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r = np.sqrt(self.distances_squared(x=x,z=z)) / self.lengthscale
        K = self.variance * np.exp( -r )
        K = self._apply_children(K, x, z)
        return K


class Matern32(Stationary):
    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(Matern32, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r = np.sqrt(self.distances_squared(x=x,z=z)) / self.lengthscale
        K = self.variance * (1.+np.sqrt(3.)*r) * np.exp(-np.sqrt(3.)*r)
        K = self._apply_children(K, x, z)
        return K


class Matern52(Stationary):
    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(Matern52, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r2 = self.distances_squared(x=x,z=z) / self.lengthscale**2
        r = np.sqrt(r2)
        K = self.variance * (1.+np.sqrt(5.)*r+(5./3)*r2) * np.exp(-np.sqrt(5.)*r)
        K = self._apply_children(K, x, z)
        return K


class GridKernel(object):
    """ simple wrapper for a kernel for GridRegression which is a product of 1d kernels """
    def __init__(self, kern_list, radial_kernel=False):
        """
        Kernel for gridded inducing point methods and structured problems

        Inputs:
            kern_list : list or 1d array of kernels
            radial_kernel : bool
                if true then will use the same kernel along each dimension. Will just grab the kernel from the first dimension to use for all.
        """
        # initialize the kernel list
        self.kern_list = kern_list

        # add the dimension of the grid
        self.grid_dim = len(kern_list)

        # check if radial kernel
        assert isinstance(radial_kernel, bool)
        self.radial_kernel = radial_kernel
        if self.radial_kernel:
            for kern in self.kern_list:
                assert kern.n_dims == self.kern_list[0].n_dims, "number of grid dims must be equal for all slices"
            self.kern_list = [self.kern_list[0],]*np.size(kern_list) # repeat the first kernel along all dimensions
        else:
            # set the variance as fixed for all but the first kernel. 
            # ... this should be valid even when the kernel is eg. a sum of other kernels
            for i in range(1,self.grid_dim):
                if hasattr(self.kern_list[i], 'fix_variance'):
                    self.kern_list[i].fix_variance()
                elif np.size(self.kern_list[i].constraint_map['variance']) > 1:
                    _logger.info("Multiple variance parameters found in the kernel, will only fix the first")
                    self.kern_list[i].constraint_map['variance'][0] = 'fixed'
                else:
                    self.kern_list[i].constraint_map['variance'] = 'fixed'

        # the the total number of dims
        self.n_dims = np.sum([kern.n_dims for kern in self.kern_list])
        return


    def cov_grid(self, x, z=None, dim_noise_var=None, use_toeplitz=False):
        """
        generates a matrix which creates a covariance matrix mapping between x1 and x2.
        Inputs:
          x : numpy.ndarray of shape (self.grid_dim,)
          z : (optional) numpy.ndarray of shape (self.grid_dim,) if None will assume x2=x1
              for both x1 and x2:
              the ith element in the array must be a matrix of size [n_mesh_i,n_dims_i]
              where n_dims_i is the number of dimensions in the ith kronecker pdt
              matrix and n_mesh_i is the number of points along the ith dimension
              of the grid.
              Note that for spatial temporal datasets, n_dims_i is probably 1
              but for other problems this might be of much higher dimensions.
          dim_noise_var : float (optional)
              diagonal term to use to shift the diagonal of each dimension to improve conditioning

        Outputs:
          K : gp_grief.tensors.KronMatrix of size determined by x and z (prod(n_mesh1(:)), prod(n_mesh2(:))
              covariance matrix
        """
        assert dim_noise_var is not None, "dim_noise_var must be specified"
        # toeplitz stuff
        if isinstance(use_toeplitz, bool):
            use_toeplitz = [use_toeplitz,] * self.grid_dim
        else:
            assert np.size(use_toeplitz) == self.grid_dim
        if np.any(use_toeplitz):
            assert z is None, "toeplitz can only be used where the (square) covariance matrix is being computed"

        # check inputs (minimal here, rest will be taken care of by calls to kern.cov)
        assert len(x) == self.grid_dim # ensure the first dimension is the same as the grid dim
        if z is None:
            cross_cov = False
            z = [None,] * self.grid_dim # array of None
        else:
            cross_cov = True
            assert len(z) == self.grid_dim # ensure the first dimension is the same as the grid dim

        # get the 1d covariance matricies
        K = []
        for i,(kern, toeplitz) in enumerate(zip(self.kern_list, use_toeplitz)): # loop through and generate the covariance matricies
            if toeplitz and z[i] is None:
                K.append(kern.cov_toeplitz(x=x[i]))
            else:
                K.append(kern.cov(x=x[i],z=z[i]))

        # now create a KronMatrix instance
        K = KronMatrix(K[::-1], sym=(z[0] is None)) # reverse the order and set as symmetric only if the two lists are identical

        # shift the diagonal of the sub-matricies if required
         # TODO: really this shouldn't just be where the diagonal is for cov matricies but anywhere the covariance is evaluated btwn two identical points
        if dim_noise_var != 0.:
            assert not cross_cov, "not implemented for cross covariance matricies yet"
            K = K.sub_shift(shift=dim_noise_var)
        return K


    def cov(self,x,z=None, dim_noise_var=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        assert dim_noise_var is None, "currenly no way to add dim_noise_var to this"

        # loop through each dimension, compute the 1(ish)-dimensional covariance and perform hadamard product
        i_cur = 0
        zi = None # set default value
        for i,kern in enumerate(self.kern_list):
            xi = x[:,i_cur:(i_cur+kern.n_dims)] # just grab a subset of the dimensions
            if z is not None:
                zi = z[:,i_cur:(i_cur+kern.n_dims)]
            i_cur += kern.n_dims

            # compute the covaraince of the subset of dimensions and multipy with the other dimensions
            if i == 0:
                K = kern.cov(x=xi,z=zi)
            else: # perform hadamard product
                K = np.multiply(K, kern.cov(x=xi,z=zi))
        return K


    def cov_kr(self,x,z, dim_noise_var=None, form_kr=True):
        """
        Evaluate covariance kernel at points to form a covariance matrix in row partitioned Khatri-Rao form

        Inputs:
            x : array of shape (N, d)
            z : numpy.ndarray of shape (d,)
              the ith element in the array must be a matrix of size [n_mesh_i,1]
              where n_mesh_i is the number of points along the ith dimension
              of the grid.
            form_kr : if True will form the KhatriRao matrix, else will just return a list of arrays

        Outputs:
            k : row partitioned Khatri-Rao matrix of shape (N, prod(n_mesh))
        """
        assert dim_noise_var is None, "currenly no way to add dim_noise_var to this"
        (N,d) = x.shape
        assert self.grid_dim == d, "currently this only works for 1-dimensional grids"

        # loop through each dimension and compute the 1-dimensional covariance matricies
        # and compute the covaraince of the subset of dimensions
        Kxz = [kern.cov(x=x[:,(i,)],z=z[i]) for i,kern in enumerate(self.kern_list)]

        # flip the order
        Kxz = Kxz[::-1]

        # convert to a Khatri-Rao Matrix
        if form_kr:
            Kxz = KhatriRaoMatrix(A=Kxz, partition=0) # row partitioned
        return Kxz

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        if self.radial_kernel:
            parameters = np.ravel(self.kern_list[0].parameters)
        else:
            parameters = np.concatenate([np.ravel(kern.parameters) for kern in self.kern_list], axis=0)
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d

        # set the parameters
        if self.radial_kernel:
            self.kern_list[0].parameters = value
            self.kern_list = [self.kern_list[0],]*np.size(self.kern_list) # repeat the first kernel along all dimensions
        else:
            i0 = 0 # counter of current position in value
            for kern in self.kern_list:
                # get the old parameters to check the size
                old = kern.parameters
                # set the parameters
                kern.parameters = value[i0:i0+np.size(old)].reshape(np.shape(old))
                i0 += np.size(old) # increment counter

    @property
    def constraints(self):
        """
        returns the kernel parameters' constraints as a 1d array
        """
        if self.radial_kernel:
            constraints = np.ravel(self.kern_list[0].constraints)
        else:
            constraints = np.concatenate([np.ravel(kern.constraints) for kern in self.kern_list], axis=0)
        return constraints

    @property
    def diag_val(self):
        """ return diagonal value of covariance matrix. Note that it's assumed the kernel is stationary """
        return self.cov(np.zeros((1,self.n_dims))).squeeze()


    def __str__(self):
        """ prints the kernel """
        s = '\nGridKernel'
        if self.radial_kernel:
            s += " Radial (same kern along all dimensions)\n"
            s += str(self.kern_list[0]) + '\n'
        else:
            for i,child in enumerate(self.kern_list):
                s += '\nGrid Dimension %d' % i
                s += str(child) + '\n'
        return s


class GriefKernel(GridKernel):
    """ kernel composed of grid-structured eigenfunctions """
    def __init__(self, kern_list, grid, n_eigs=1000, reweight_eig_funs=True, opt_kernel_params=False, w=1., dim_noise_var=1e-12, log_KRrowcol=True, **kwargs):
        """
        Inputs:
            kern_list : list of 1d kernels
            grid : inducing point grid (e.g. `grid=gp_grief.grid.InducingGrid(x)` where x is the train inputs)
            n_eigs : number of eigenvalues to use
            reweight_eig_funs : whether the eigenfunctions should be reweighted
            opt_kernel_params : whether the kernel hyperparameters should be optimized
            w : initial basis function weights (default is unity)
        """
        self.reweight_eig_funs = bool(reweight_eig_funs)
        self.opt_kernel_params = bool(opt_kernel_params)
        super(GriefKernel, self).__init__(kern_list=kern_list, **kwargs)
        assert isinstance(grid,InducingGrid), "must be an InducingGrid"
        assert grid.input_dim == self.n_dims, "training set number of dimensions don't match the grid provided"
        self.grid = grid
        self.dim_noise_var = float(dim_noise_var)
        self.n_eigs = int(min(n_eigs, self.grid.num_data)) # number of eigenvalues to use for the rank reduced approximation

        # set the contraints for the base kernel hyperparameters
        if not self.opt_kernel_params: # then fix everything
            for i,kern in enumerate(self.kern_list):
                if isinstance(kern, GPyKernel):
                    self.kern_list[i].constraint_list = np.tile('fixed', np.shape(kern.constraint_list))
                else:
                    for key in kern.constraint_map:
                        self.kern_list[i].constraint_map[key] = np.tile('fixed', np.shape(kern.constraint_map[key]))

        # set the constraints for the weights
        if self.reweight_eig_funs:
            self.w_constraints = np.array(['+ve',] * self.n_eigs, dtype='|S10')
        else:
            self.w_constraints = np.array(['fixed',] * self.n_eigs, dtype='|S10')


        if w == 1.:
            self.w = np.ones(self.n_eigs)
        else:
            assert w.shape == (self.n_eigs,)
            assert np.all(w > 0.), "w's must be positive"
            self.w = w

        # initialize some stuff
        self._old_base_kern_params = None
        self.log_KRrowcol = log_KRrowcol


    def cov(self, x, z=None):
        """
        returns everything needed for the covariance matrix and its inverse, etc.
        Note that z should generally not be specified, you can save work by just computing
        Phi_L or Phi_R, see how the computation is done below.

        Outputs:
            Phi_L : left basis function or coefficient matrix
            w : basis function weights
            Phi_R : right basis function or coefficient matrix (if z is none then will be Phi_L)

        Notes:
            ```
            from scipy.sparse import dia_matrix
            from gp_grief.tensors import TensorProduct
            K = TensorProduct([Phi_L, dia_matrix((w/lam, 0),shape=(w.size,)*2), Phi_R.T])
            ```
        """
        assert x.shape[1] == self.n_dims
        if z is not None: # then computing cross cov matrix
            Phi_L = self.cov(x=x)[0]
            Phi_R = self.cov(x=z)[0]
        else:
            # setup inducing covariance matrix and eigenvals/vecs
            self._setup_inducing_cov()

            # compute the left coefficient matrix
            # first get the cross covariance matrix
            Kxu = super(GriefKernel,self).cov_kr(x=x,z=self.grid.xg, form_kr=False)
            Kux = [k.T for k in Kxu]

            # form the RowColKhatriRaoMatrix 
            if self.log_KRrowcol: # form and rescale in a numerically stable manner
                log_matrix, sign = expand_SKC(S=self._Sp, K=self._Quu.T.K, C=Kux, logged=True)
                Phi_L = sign.T * np.exp(log_matrix.T - 0.5*self._log_lam.reshape((1,-1)))
            else:
                Phi_L = expand_SKC(S=self._Sp, K=self._Quu.T.K, C=Kux, logged=False).T / np.sqrt(np.exp(self._log_lam.reshape((1,-1))))

            # compute the left coefficient matrix (which is identical)
            Phi_R = Phi_L
        return Phi_L, self.w, Phi_R

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        # first get the regular parameters
        parameters = super(GriefKernel, self).parameters
        # then add the eigenfunction weights
        parameters = np.concatenate([parameters, self.w], axis=0)
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        # get the number of base kernel hyperparameters
        n_theta = value.size-self.n_eigs
        # set the parameters of the base kernel
        super(GriefKernel, self.__class__).parameters.fset(self, value[:n_theta]) # call setter in the super method, this
        # now set the weights
        self.w = value[n_theta:]

    @property
    def constraints(self):
        """
        returns the kernel parameters' constraints as a 1d array
        """
        # first get the regular constraints
        constraints = super(GriefKernel, self).constraints
        # then add the eigenfunction weights if nessessary
        constraints = np.concatenate([constraints, self.w_constraints], axis=0)
        return constraints

    @property
    def diag_val(self):
        """ return diagonal value of covariance matrix. Note that it's assumed the kernel is stationary """
        raise NotImplementedError('')


    def _setup_inducing_cov(self):
        """ setup the covariance matrix on the inducing grid, factorize and find largest eigvals/vecs """
        # determine if anything needs to be recomputed
        base_kern_params = super(GriefKernel, self).parameters
        if self._old_base_kern_params is not None and np.array_equal(self._old_base_kern_params, base_kern_params):
            return # then no need to recompute

        # get the covariance matrix on the grid
        Kuu = self.cov_grid(self.grid.xg, dim_noise_var=self.dim_noise_var)

        # compute svd of Kuu
        (self._Quu,T) = Kuu.schur()
        all_eig_vals = T.diag()

        # get the biggest eigenvalues and eigenvectors
        n_eigs = int(min(self.n_eigs, all_eig_vals.shape[0])) # can't use more eigenvalues then all of them
        eig_pos, self._log_lam = all_eig_vals.find_extremum_eigs(n_eigs=n_eigs,mode='largest',log_expand=True)[:2]

        # create a Khatri-Rao selection matrix, Sp 
        self._Sp = [SelectionMatrixSparse((col, Kuu.K[i].shape[0])) for i,col in enumerate(eig_pos.T)]

        # save the parameters
        self._old_base_kern_params = base_kern_params


    def __str__(self):
        """ prints the kernel """
        s = '\nGriefKernel'
        if self.radial_kernel:
            s += " Radial (same kern along all dimensions)\n"
            s += str(self.kern_list[0]) + '\n'
        else:
            for i,child in enumerate(self.kern_list):
                s += '\nGrid Dimension %d' % i
                s += str(child) + '\n'

        # now print the weights
        from tabulate import tabulate
        s += "Eigenfunction Weights:\n"
        s += str(tabulate([["weight %03d"%i, w, wm_lam, constraint]
                           for i,(w, wm_lam, constraint) in enumerate(zip(self.w, np.exp(np.log(self.w) - self._log_lam + np.log(self.grid.num_data)), self.w_constraints))],
                          headers=['Name', 'w Value', 'w*m/lam Value', 'Constraint'], tablefmt='orgtbl'))
        # Note that we multiply w/lam times m because the inner products involved in the eigenfunctions sum up huge vectors of length m so the quotient would be washed out without this
        s += '\n'
        return s


class WEBKernel(object):
    """ simple class for parametrizing the weighted basis function kernel """

    def __init__(self, initial_weights):
        assert isinstance(initial_weights, np.ndarray)
        assert np.ndim(initial_weights) == 1
        self.p = np.size(initial_weights)
        self.parameters = initial_weights
        self.constraints = ['+ve',]*self.p


class RBF_RFF(object):
    """ random fourier features for an RBF kernel """
    def __init__(self, d, log_lengthscale=0, n_rffs=1000, dtype=np.float64, tune_len=True):
        """
        squared exponential kernel

        Input:
            d : number of input dims
            n_rffs : number of random features (will actually used twice this value)
        """
        # TODO: add ability to be non ARD
        logger.info("initializing RBF kernel")
        self.d = int(d)
        self.n_rffs = int(n_rffs)
        self.n_features = 2*n_rffs # each random feature is broken into two
        self.dtype = dtype
        self.freq_weights = np.asarray(np.random.normal(size=(self.d, self.n_rffs), loc=0, scale=1.), dtype=self.dtype)
        self.bf_scale = 1./np.sqrt(self.n_rffs)

        # Set the lengthscale variable
        if np.size(log_lengthscale)==1 and log_lengthscale == 0:
            log_lengthscale = np.zeros((d,1), dtype=self.dtype)
        else:
            log_lengthscale = np.asarray(log_lengthscale, dtype=self.dtype).reshape((d,1))
        self.log_ell = log_lengthscale


    def Phi(self, x):
        """
        Get the basis function matrix

        Inputs:
            x : (n, d) input postions

        Outputs:
            Phi : (n, 2*n_features)
        """
        Xfreq = np.dot(x, self.freq_weights/np.exp(self.log_ell)) # scale the frequencies by the lengthscale and multiply with the inputs
        return self.bf_scale * np.concatenate([np.cos(Xfreq), np.sin(Xfreq)], axis=1)


