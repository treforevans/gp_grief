from .tensors import KhatriRaoMatrix, BlockMatrix
from .linalg import uniquetol
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
import logging
import warnings
from pdb import set_trace
logger = logging.getLogger(__name__)


def nd_grid(*xg):
    """
    This mimics the behaviour of nd_grid in matlab.
    (np.mgrid behaves similarly however I don't get how to call it clearly.)
    """
    grid_shape = [np.shape(xg1d)[0] for xg1d in xg] # shape of the grid
    d = np.size(grid_shape)
    N = np.product(grid_shape)
    X_mesh = np.empty(d, dtype=object)
    for i, xg1d in enumerate(xg): # for each 1d component
        if np.ndim(xg1d) > 1:
            assert np.shape(xg1d)[1] == 1, "only currently support each grid dimension being 1d"
        n = np.shape(xg1d)[0] # number of points along dimension of grid
        slice_shape = np.ones(d, dtype=int);   slice_shape[i] = n # shape of the slice where xg1d fits
        stack_shape = np.copy(grid_shape);     stack_shape[i] = 1 # shape of how the slice should be tiled
        X_mesh[i] = np.tile(xg1d.reshape(slice_shape), stack_shape) # this is the single dimension on the full grid
    return X_mesh


def grid2mat(*xg):
    """
    transforms a grid to a numpy matrix of points

    Inputs:
        *xg : ith input is a 1D array of the grid locations along the ith dimension

    Outputs:
        x : numpy matrix of size (N,d) where N = n1*n2*...*nd = len(xg[0])*len(xg[1])*...*len(xg[d-1])
    """
    X_mesh = nd_grid(*xg) # this is the meshgrid, all I have to do is flatten it
    d = X_mesh.shape[0]
    N = X_mesh[0].size
    x = np.zeros((N, d)) # initialize
    for i, X1d in enumerate(X_mesh): # for each 1d component of the mesh
        x[:,i] = X1d.reshape(-1, order='C') # reshape it into a vector
    return x


class InducingGrid(object):
    """ inducing point grid for structured kernel interpolation """

    def __init__(self, x=None, mbar=10, eq=True, to_plot=False, mbar_min=1, xg=None, beyond_domain=None):
        """
        Generates an inducing point grid.

        Inputs to generate an inducing grid from scattered data:
            x : (n_train,n_dims) scattered training points
            mbar : if mbar > 1 then indicates the number of points desired along each dim. Else if in (0,1] then
                the number of points will be that fraction of unique values.
                if mbar is a tuple, then the value applys for that specific dimension
            eq : if true forces evenly spaced grid, else will use a kmeans algorithm which tries to
                 get an even number of points closest to each inducing point.
            to_plot : if true then plot statistics of grid. (default false)
                if true then will plot to screen. if a string then will plot to the filename specified
            mbar_min : the minimum number of points per dimension. Note that k will be changed to be no
                greater than the number of unique points per dimension if not eq. This parameter sets a lower
                bound for this value which is useful for eg. when SKI is employed.
            beyond_domain : None or float
                if None, then no effect. If a value then will go that fraction beyond bounds along each
                dimension. Note that the number of points specifed won't be violated and a point will still
                be placed right on the edge of the bounds if eq=True

        Inputs to generate an instance from a user specified grid:
            xg : list specifying the grid in each dimension.
                each element in xg must be the array of points along each dimension of the grid. eg.
                    xg[i].shape = (grid_shape[i], grid_sub_dim[i])
                Points along each grid dimension should be sorted ascending if the demension is 1d.
                No other inputs are nessessary and if specified then they'll be ignored
        """
        logger.debug('Initializing inducing grid.')
        k = mbar; del mbar # mbar is an alias
        k_min = mbar_min; del mbar_min # mbar_min is an alias
        if xg is None: # then generate a grid from the scattered points x
            # deal with inputs
            assert isinstance(x,np.ndarray)
            assert x.ndim == 2
            self.eq = eq
            if not isinstance(k,(tuple,list,np.ndarray)):
                k = (k,)*x.shape[1]

            # get some statistics and counts (just assuming 1d along each dimension)
            (n_train, self.grid_dim) = x.shape # number of training points, number of grid dimensions
            self.grid_sub_dim = np.ones(self.grid_dim, dtype=int) # number of sub dimensions along each grid dim
            self.input_dim = np.sum(self.grid_sub_dim) # total number of dimensions
            self.grid_shape = np.zeros(self.grid_dim, dtype=int); # number of points along each sub dimension
            x_rng = np.vstack((np.amin(x,axis=0), np.amax(x,axis=0), np.ptp(x,axis=0))).T
            n_unq = np.array([np.unique(x[:,i]).size for i in range(self.grid_dim)])
            if not np.all(n_unq >= 2):
                logger.debug('some dimension have < 2 unique points')
            for i,ki in enumerate(k):
                if ki <= 1:
                    self.grid_shape[i] = np.int32(np.maximum(np.ceil(ki*n_unq[i]),k_min));
                else:
                    assert np.mod(ki,1) == 0, "if k > 1 then it must be an integer"
                    # don't allow the number of points to be greater than n_unq
                    self.grid_shape[i] = np.int32(np.maximum(np.minimum(ki, n_unq[i]), k_min));
            self.num_data = np.prod(np.float64(self.grid_shape)) # total number of points on the full grid

            # check if bounds are to be added, in which case I want to call recursively
            if beyond_domain is not None:
                assert np.all(self.grid_shape >= 2), "bounds need at least 2 points per dim"
                # get the grid with no bounds but 2 less points per dimension
                xg = InducingGrid(x=x, k=self.grid_shape-2, eq=eq, to_plot=False, k_min=0, xg=None, beyond_domain=None).xg
                for i in range(x.shape[1]):
                    xg[i] = np.vstack((x_rng[i,0]-beyond_domain*x_rng[i,2], xg[i], x_rng[i,1]+beyond_domain*x_rng[i,2])) # add the points that go beyond domain
                # since xg is now specified, it will be added to the grid below
            else:
                #figure out if the grid should be on unique points
                on_unique = self.grid_shape == n_unq # whether or not the grid is exactly on unique values

                # create the grid
                # self.xg is a list of length n_dims which specifies the grid along each dimension.
                self.xg = np.empty(self.grid_dim, dtype=object)
                for i_d in range(self.grid_dim):
                    if on_unique[i_d]: # then place the grid on the unique values
                        self.xg[i_d] = np.unique(x[:,i_d]).reshape((-1,1))
                    elif self.eq: # equally spaced grid points
                        self.xg[i_d] = np.linspace(x_rng[i_d,0],x_rng[i_d,1],num=self.grid_shape[i_d]).reshape((-1,1))
                    elif self.grid_shape[i_d] == 2: # then just place on the ends
                        self.xg[i_d] = x_rng[i_d,:2].reshape((-1,1))
                    else: # non equally spaced grid points
                        """
                        do a two-pronged kmeans clustering strategy where you find clusters of clusters:
                            1) indentify clusters in the data, I don't want to reconsider points in the same cluster twice
                            1.5) filter any clusters which are close together
                            2) rerun kmeans using the cluster centers to get the grid points
                            2.5) filter any nodes which are close together
                        This makes clusters which aren't too close together and also encourages spread throughout the space
                        """
                        # TODO: it seems that it's actually important to bound clusters, not just have them nearby
                        #    I can try to implement this maybe

                        node_tol = x_rng[i_d,2]/(3*self.grid_shape[i_d])
                        # 1) identify clusters in x. Use more than the final number of grid points
                        x_clusters = MiniBatchKMeans( # will be faster for large problems
                            n_clusters=np.minimum(3*self.grid_shape[i_d],n_unq[i_d]), n_init=1, max_iter=100, tol=0.001,
                        ).fit(np.unique(x[:,i_d]).reshape((-1,1)) # I don't want to recount duplicates more than once
                             ).cluster_centers_.reshape((-1,1))

                        # 1.5) remove clusters which are close together
                        x_clusters = uniquetol(x_clusters.squeeze(),
                                               tol=node_tol/2, # set a loose tol here
                                               ).reshape((-1,1))
                        self.grid_shape[i_d] = np.minimum(x_clusters.size, self.grid_shape[i_d])

                        if self.grid_shape[i_d] == x_clusters.size: # then place the nodes on the clusters
                            self.xg[i_d] = x_clusters
                        elif self.grid_shape[i_d] > 2: # perform the second kmeans clustering
                            # 2) get the final grid points
                            self.xg[i_d] = KMeans(
                                n_clusters=self.grid_shape[i_d]-2, n_init=1, max_iter=100, tol=0.001, verbose=False,
                            ).fit(np.vstack((x_rng[i_d,0], x_clusters, x_rng[i_d,1])) # add the extreme values back to bias the nodes
                                 ).cluster_centers_.reshape((-1,1))

                            # 2.5) remove nodes which are close together
                            self.xg[i_d] = uniquetol(self.xg[i_d].squeeze(), tol=node_tol).reshape((-1,1))
                        else: # initiaze empty grid, extreme values will be added later
                            self.xg[i_d] = np.zeros((0,1))

                        # sort the inducing points and place nodes at the extreme values
                        self.xg[i_d].sort(axis=0)
                        self.xg[i_d] = np.vstack((x_rng[i_d,0],self.xg[i_d],x_rng[i_d,1]))
                        if np.abs(self.xg[i_d][1,0] - self.xg[i_d][0,0]) < node_tol: #check if too close together at ends
                            self.xg[i_d] = np.delete(self.xg[i_d],1,axis=0)
                        if np.abs(self.xg[i_d][-1,0] - self.xg[i_d][-2,0]) < node_tol: #check if too close together at ends
                            self.xg[i_d] = np.delete(self.xg[i_d],-2,axis=0)
                        assert x_rng[i_d,0] == self.xg[i_d][0,0] and x_rng[i_d,1] == self.xg[i_d][-1,0], "extremum values didn't make it into set"
                        self.grid_shape[i_d] = self.xg[i_d].size
        if xg is not None: # a grid has already been specified so use this instead
            self.xg        = np.asarray(xg)
            self.grid_dim = self.xg.shape[0] # number of grid dimensions
            self.grid_shape   = np.zeros(self.grid_dim, dtype=int) # number of points along each sub dimension
            self.grid_sub_dim = np.zeros(self.grid_dim, dtype=int) # number of sub dimensions along each grid dim
            for i,X in enumerate(self.xg): # loop over grid dimensions
                assert X.ndim == 2, "each element in xg must be a 2d array"
                self.grid_sub_dim[i] = X.shape[1]
                self.grid_shape[i]   = X.shape[0]
            self.input_dim = np.sum(self.grid_sub_dim) # total number of dimensions
            self.num_data = np.prod(np.float64(self.grid_shape)) # total number of points on the full grid
            self.eq = None

        # plot the grid
        if to_plot is True:
            self.plot(x)
        elif isinstance(to_plot, str):
            self.plot(x, fname=to_plot)


    def plot(self, x, fname=None):
        """
        plots the grid along each dimension along with a frequency histogram of the location points x
        """
        logger.debug('plotting inducing grid')
        assert np.all(self.grid_sub_dim == 1), "only works when grid dims are 1d"
        import matplotlib
        if fname is not None:
            matplotlib.use('Agg')
            from matplotlib.backends.backend_pdf import PdfPages
        matplotlib.rcParams.update({'font.size': 10})
        import matplotlib.pyplot as plt
        def plot1dim(i_dim):
            """ function for ploting one grid dimension"""
            freq = plt.hist(x=x[:,i_dim], bins=min(100,4*self.grid_shape[i_dim]))[0]
            plt.plot(self.xg[i_dim][:,0],np.zeros(self.grid_shape[i_dim]) + 0.5*np.max(freq),'ko',markersize=3)
            plt.xlabel(r'x_%d'%i_dim)
            plt.ylabel('Frequency')
            plt.title('Dim %d, m = %d' % (i_dim, self.grid_shape[i_dim]))

        # now loop through and plot
        if fname is None: # then plot to screen
            plt.figure(figsize=(4,4*self.grid_dim))
            for i in range(self.grid_dim):
                plt.subplot(self.grid_dim,1,i+1)
                plot1dim(i_dim=i)
            plt.suptitle('Inducing Grid and Train Point Dist.')
            plt.show(block=False)
        else: # save to file
            with PdfPages(fname) as pdf:
                for i in range(self.grid_dim):
                    plt.figure(figsize=(4,4))
                    plot1dim(i_dim=i)
                    pdf.savefig()
                    plt.close()


    def __getitem__(self, key):
        """ so you can get self[key] """
        return self.xg[key]


    def __setitem__(self, key, value):
        """ so you can set self[key] = value """
        self.xg[key] = value


