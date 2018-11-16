from .kern import BaseKernel, GridKernel, GriefKernel, WEBKernel
from .linalg import solve_schur, solve_chol, solver_counter, LogexpTransformation
from .grid import InducingGrid
from .stats import norm, lognorm, RandomVariable, StreamMeanVar
from pymcmc import Model as pymcmc_model

# numpy/scipy stuff
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import fmin_l_bfgs_b

# development stuff
from numpy.linalg.linalg import LinAlgError
from numpy.testing import assert_array_almost_equal
from traceback import format_exc
from pdb import set_trace
from logging import getLogger
from warnings import warn
logger = getLogger(__name__)

class BaseModel(object):
    param_shift = {'+ve':1e-200, '-ve':-1e-200} # for positive or negative constrained problems, this is as close as it can get to zero, by default this is not used
    _transformations = {'+ve':LogexpTransformation()}

    def __init__(self):
        """ initialize a few instance variables """
        logger.debug('Initializing %s model.' % self.__class__.__name__)
        self.dependent_attributes = ['_alpha','_log_like','_gradient','_K','_log_det'] # these are set to None whenever parameters are updated
        self._previous_parameters = None # previous parameters from last call to parameters property
        self.grad_method = None # could be {'finite_difference','adjoint'}
        self.noise_var_constraint = '+ve' # constraint for the Gaussian noise variance
        return


    def log_likelihood(self, return_gradient=False):
        """
        computes the log likelihood and the gradient (if hasn't already been computed).

        If the return_gradient flag is set then then the gradient will be returned as a second arguement.
        """
        p = self.parameters # this has to be called first to fetch parameters and ensure internal consistency

        # check if I need to recompute anything
        if return_gradient and (self._gradient is None):
            # compute the log likelihood and gradient wrt the parameters
            if 'adjoint' in self.grad_method:
                (self._log_like, self._gradient) = self._adjoint_gradient(p)
            elif 'finite_difference' in self.grad_method:
                (self._log_like, self._gradient) = self._finite_diff_gradient(p)
            else:
                raise RuntimeError('unknown grad_method %s' % repr(self.grad_method))
        elif self._log_like is None: # just compute the log-likelihood without the gradient
            self._log_like = self._compute_log_likelihood(p)
        else: # everything is already computed
            pass

        if return_gradient: # return both
            return self._log_like, self._gradient
        else: # just return likelihood
            return self._log_like


    def optimize(self, max_iters=1000, messages=True, use_counter=False, factr=10000000.0, pgtol=1e-05):
        """
        maximize the log likelihood

        Inputs:
            max_iters : int
                maximum number of optimization iterations
            factr, pgtol : lbfgsb convergence criteria, see fmin_l_bfgs_b help for more details
                use factr of 1e12 for low accuracy, 10 for extremely high accuracy (default 1e7)
        """
        logger.debug('Beginning MLE to optimize hyperparameters. grad_method=%s' % self.grad_method)

        # setup the optimization
        try:
            x0 = self._transform_parameters(self.parameters) # get the transformed value to start at
            assert np.all(np.isfinite(x0)), "initial transformation led to non-finite values"
        except:
            logger.error('Transformation failed for initial values. Ensure constraints are met or the value is not too small.')
            raise

        # filter out the fixed parameters
        free = np.logical_not(self._fixed_indicies)
        x0 = x0[free]

        # setup the counter
        if use_counter:
            self._counter = solver_counter(disp=True)
        else:
            self._counter = None

        # run the optimization
        try:
            x_opt, f_opt, opt = fmin_l_bfgs_b(func=self._objective_grad, x0=x0, factr=factr, pgtol=pgtol, maxiter=max_iters, disp=messages)
        except (KeyboardInterrupt,IndexError): # sometimes interrupting gives index error for scipy sparse matricies it seems
            logger.info('Keyboard interrupt raised. Cleaning up...')
            if self._counter is not None and self._counter.backup is not None:# use the backed up copy of parameters from the last iteration
                self.parameters = self._counter.backup[1]
                logger.info('will return best parameter set with log-likelihood = %.4g' % self._counter.backup[0])
        else:
            logger.info('Function Evals: %d. Exit status: %s' % (f_opt, opt['warnflag']))
            # extract the optimal value and set the parameters to this
            transformed_parameters = self._previous_parameters # the default parameters are the previous ones
            transformed_parameters[free] = x_opt # these are the transformed optimal parameters
            self.parameters = self._untransform_parameters(transformed_parameters) # untransform
        return opt


    def checkgrad(self, decimal=3, raise_if_fails=True):
        """
        checks the gradient and raises if does not pass
        """
        grad_exact = self._finite_diff_gradient(self.parameters)[1]
        grad_exact[self._fixed_indicies] = 1 # I don't care about the gradients of fixed variables
        grad_analytic = self.log_likelihood(return_gradient=True)[1]
        grad_analytic[self._fixed_indicies] = 1 # I don't care about the gradients of fixed variables

        # first protect from nan values incase both analytic and exact are small
        protected_nan = np.logical_and(np.abs(grad_exact) < 1e-8, np.abs(grad_analytic) < 1e-8)

        # now protect against division by zero. I do this by just removing the values that have a small absoute error
        # since if the absolute error is tiny then I don't really care about relative error
        protected_div0 = np.abs(grad_exact-grad_analytic) < 1e-5

        # now artificially change these protected values
        grad_exact[np.logical_or(protected_nan, protected_div0)] = 1.
        grad_analytic[np.logical_or(protected_nan, protected_div0)] = 1.

        try:
            assert_array_almost_equal(grad_exact / grad_analytic, np.ones(grad_exact.shape),
                                      decimal=decimal, err_msg='Gradient ratio did not meet tolerance.')
        except:
            logger.info('Gradient check failed.')
            logger.debug('[[Finite-Difference Gradient], [Analytical Gradient]]:\n%s\n' % repr(np.asarray([grad_exact,grad_analytic])))
            if raise_if_fails:
                raise
            else:
                logger.info(format_exc()) # print the output
                return False
        else:
            logger.info('Gradient check passed.')
            return True

    @property
    def parameters(self):
        """
        this gets the parameters from the object attributes
        """
        parameters = np.concatenate((np.ravel(self.noise_var), self.kern.parameters),axis=0)

        # check if the parameters have changed
        if not np.array_equal(parameters, self._previous_parameters):
            # remove the internal variables that rely on the parameters
            for attr in self.dependent_attributes:
                setattr(self, attr, None)
            # update the previous parameter array
            self._previous_parameters = parameters.copy()
        return parameters.copy()

    @parameters.setter
    def parameters(self,parameters):
        """
        this takes the optimization variables parameters and sets the internal state of self
        to make it consistent with the variables
        """
        # set the parameters internally
        self.noise_var       = parameters[0]
        self.kern.parameters = parameters[1:]

        # check if the parameters have changed
        if not np.array_equal(parameters, self._previous_parameters):
            # remove the internal variables that rely on the parameters
            for attr in self.dependent_attributes:
                setattr(self, attr, None)
            # update the previous parameter array
            self._previous_parameters = parameters.copy()
        return parameters

    @property
    def constraints(self):
        """ returns the model parameter constraints as a list """
        constraints = np.concatenate((np.ravel(self.noise_var_constraint), self.kern.constraints),axis=0)
        return constraints


    def predict(self,Xnew,compute_var=None):
        """
        make predictions at new points

        MUST begin with a call to parameters property to ensure internal state consistent
        """
        raise NotImplementedError('')


    def fit(self):
        """
        determines the weight vector _alpha

        MUST begin with a call to parameters property to ensure internal state consistent
        """
        raise NotImplementedError('')


    def _objective_grad(self,transformed_free_parameters):
        """ determines the objective and gradients in the transformed input space """
        # get the fixed indices and add to the transformed parameters
        free = np.logical_not(self._fixed_indicies)
        transformed_parameters = self._previous_parameters # the default parameters are the previous ones
        transformed_parameters[free] = transformed_free_parameters
        try:
            # untransform and internalize parameters
            self.parameters = self._untransform_parameters(transformed_parameters)
            # compute objective and gradient in untransformed space
            (objective, gradient) = self.log_likelihood(return_gradient=True)
            objective = -objective # since we want to minimize
            gradient =  -gradient
            # ensure the values are finite
            if not np.isfinite(objective):
                logger.debug('objective is not finite')
            if not np.all(np.isfinite(gradient[free])):
                logger.debug('some derivatives are non-finite')
            # transform the gradient 
            gradient = self._transform_gradient(self.parameters, gradient)
        except (LinAlgError, ZeroDivisionError, ValueError):
            logger.error('numerical issue while computing the log-likelihood or gradient.')
            logger.debug('Here is the current model where the failure occured:\n' + self.__str__())
            raise
        # get rid of the gradients of the fixed parameters
        free_gradient = gradient[free]

        # call the counter if ness
        if self._counter is not None:
            msg='log-likelihood=%.4g, gradient_norm=%.2g' % (-objective, np.linalg.norm(gradient))
            if self._counter.backup is None or self._counter.backup[0] < -objective: # then update backup
                self._counter(msg=msg,store=(-objective,self.parameters.copy()))
            else: # don't update backup
                self._counter(msg=msg)
        return objective, free_gradient

    @property
    def _fixed_indicies(self):
        """ returns a bool array specifiying where the indicies are fixed """
        fixed_inds = self.constraints == 'fixed'
        return fixed_inds

    @property
    def _free_indicies(self):
        """ returns a bool array specifiying where the indicies are free """
        return np.logical_not(self._fixed_indicies)


    def _transform_parameters(self, parameters):
        """
        applies a transformation to the parameters based on a constraint
        """
        constraints = self.constraints
        assert parameters.size == np.size(constraints) # check if sizes correct
        transformed_parameters = np.zeros(parameters.size)
        for i,(param,constraint) in enumerate(zip(parameters,constraints)):
            if constraint is None or constraint == 'fixed' or constraint == '': # then no transformation
                transformed_parameters[i] = param
            else: # I need to transform the parameters
                transformed_parameters[i] = self._transformations[constraint].transform(param - self.param_shift[constraint])

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(transformed_parameters)):
            logger.debug('transformation led to non-finite value')
        return transformed_parameters


    def _transform_gradient(self, parameters, gradients):
        """
        see _transform parameters
        """
        constraints = self.constraints
        assert parameters.size == gradients.size == np.size(constraints) # check if sizes correct
        transformed_grads      = np.zeros(parameters.size)
        for i,(param,grad,constraint) in enumerate(zip(parameters,gradients,constraints)):
            if constraint is None or constraint == '': # then no transformation
                transformed_grads[i] = grad
            elif constraint != 'fixed': # then apply a transformation (if fixed then do nothing)
                transformed_grads[i] = self._transformations[constraint].transform_grad(param - self.param_shift[constraint],grad)

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(transformed_grads)):
            logger.debug('transformation led to non-finite value')
        return transformed_grads


    def _untransform_parameters(self, transformed_parameters):
        """ applies a reverse transformation to the parameters given constraints"""
        assert transformed_parameters.size == np.size(self.constraints) # check if sizes correct
        parameters = np.zeros(transformed_parameters.size)
        for i,(t_param,constraint) in enumerate(zip(transformed_parameters,self.constraints)):
            if constraint is None or constraint == 'fixed' or constraint == '': # then no transformation
                parameters[i] = t_param
            else:
                parameters[i] = self._transformations[constraint].inverse_transform(t_param) + self.param_shift[constraint]

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(parameters)):
            logger.debug('transformation led to non-finite value')
        return parameters


    def _finite_diff_gradient(self, parameters):
        """
        helper function to compute function gradients by finite difference.

        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters
            log_like : float
                log likelihood at the current point

        Outputs:
            log_likelihood
        """
        assert isinstance(parameters,np.ndarray)
        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]

        # first take a forward step in each direction
        step = 1e-6 # finite difference step
        log_like_fs = np.zeros(free_inds.size)
        for i,param_idx in enumerate(free_inds): # loop through all free indicies
            p_fs = parameters.copy()
            p_fs[param_idx] += step # take a step forward
            log_like_fs[i] = self._compute_log_likelihood(p_fs) # compute the log likelihood at the forward step

        # compute the log likelihood at current point
        log_like = self._compute_log_likelihood(parameters)

        # compute the gradient
        gradient = np.zeros(parameters.shape) # default gradient is zero
        gradient[free_inds] = (log_like_fs-log_like) # compute the difference for the free parameters
        #if np.any(np.abs(gradient[free_inds]) < 1e-12):
            #logger.debug('difference computed during finite-difference step is too small. Results may be inaccurate.')
        gradient[free_inds] = gradient[free_inds]/step # divide by the step length
        return log_like, gradient


    def _compute_log_likelihood(self, parameters):
        """
        helper function to compute log likelihood.
        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters

        Outputs:
            log_likelihood
        """
        raise NotImplementedError('')


    def _adjoint_gradient(self,parameters):
        raise NotImplementedError('')
        return log_like, gradient


    def __str__(self):
        from tabulate import tabulate
        s = '\n%s Model\n' % self.__class__.__name__

        # print the  noise_var stuff
        s += str(tabulate([['noise_var',self.noise_var,self.noise_var_constraint]],
                          headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl')) + '\n'

        # print the kernel stuff
        s += str(self.kern)
        return s


class PyMCMC_Wrapper(pymcmc_model):
    """ wrapper class for pymcmc. All the meat happens in the params.setter property """

    def __init__(self, model, priors=None, lognorm_kwargs={}, norm_kwargs={}):
        """
        Inputs:
            priors : if specified should be a list of length of all parameters (not just those free)
        """
        assert isinstance(model, BaseModel)
        super(PyMCMC_Wrapper, self).__init__(name=self.__class__.__name__)
        self.model = model
        self.param_chain = None # this will be filled when being sampled

        # save some stuff about the parameters
        self.free = self.model._free_indicies
        self.constraints = self.model.constraints[self.free]
        self._num_params = np.count_nonzero(self.free)

        # set the priors
        if priors is None: # then need to initialize them with defaults
            lognorm_prior = lognorm(**lognorm_kwargs)
            norm_prior = norm(**norm_kwargs)
            self.priors = np.empty(self.constraints.size, dtype=object)
            for i,con in enumerate(self.constraints):
                if con is None or con is '':
                    self.priors[i] = norm_prior
                elif '+ve' in con:
                    self.priors[i] = lognorm_prior
                elif '-ve' in con:
                    self.priors[i] = lognorm_prior
                    self.priors[i].scale = -lognorm_prior.scale # the -scale will make it negative
                else:
                    raise ValueError("no default prior set for this constraint")
        else:
            assert np.size(priors) == self.constraints.size
            assert isinstance(priors[0], RandomVariable)
            self.priors = priors

        # initialize the model at its current state
        self.params = self.model._transform_parameters(self.model.parameters)[self.free]


    def __getstate__(self):
        """
        Get the state of the model.

        This shoud return the state of the model. That is, return everything
        is necessary to avoid redundant computations. It is also used in oder
        to pickle the object or send it over the network.
        """
        return self._state


    def __setstate__(self, state):
        """
        Set the state of the model.

        This is supposed to take the return value of the
        :method:`Model.__getstate__`.
        """
        self._state = state

    @property
    def num_params(self):
        """
        Return the number of parameters.
        """
        return self._num_params

    @property
    def params(self):
        """
        Set/Get the transformed parameters.
        """
        return self._state['transformed_parameters'][self.free]

    @params.setter
    def params(self, transformed_free_parameters):
        """ set parameters and update the model state """
        assert transformed_free_parameters.size == self.num_params, "only the free parameters should be passed"
        transformed_parameters = self.model._previous_parameters # the default parameters are the previous ones
        transformed_parameters[self.free] = transformed_free_parameters
        parameters = self.model._untransform_parameters(transformed_parameters)
        self.model.parameters = parameters

        # update the model state
        #logger.debug("computing for params %s" % str(parameters))
        self._state = dict(transformed_parameters=transformed_parameters, parameters=parameters)
        gradient = dict() # this is the untransformed gradient

        # compute the log likelihood
        self._state['log_likelihood'], gradient['log_likelihood'] = self.model.log_likelihood(return_gradient=True)

        # compute the log prior
        # we assume all priors are independent the log prior is just sum up the log pdf of each
        self._state['log_prior'] = np.sum([rv.logpdf(val) for rv,val in zip(self.priors, self.model.parameters[self.free])])
        gradient['log_prior'] = np.zeros(parameters.size) # initialize the gradients for all the parameters (even those fixed)
        gradient['log_prior'][self.free] = [rv.logpdf_grad(val) for rv,val in zip(self.priors, self.model.parameters[self.free])]

        # now transform the gradients
        for key, grad in gradient.iteritems():
            # ensure finite
            if not np.all(np.isfinite(grad[self.free])):
                logger.debug('some %s derivatives are non-finite' % key)

            # transform the gradient after expanding
            self._state['grad_%s_transformed'%key] = self.model._transform_gradient(parameters, grad)

    @property
    def log_likelihood(self):
        return self._state['log_likelihood']

    @property
    def log_prior(self):
        return self._state['log_prior']

    @property
    def grad_log_likelihood(self):
        return self._state['grad_log_likelihood_transformed'][self.free]

    @property
    def grad_log_prior(self):
        return self._state['grad_log_prior_transformed'][self.free]


    def predict(self, Xnew, return_samples=False, diag_cov=False):
        """
        predict at test points.

        Inputs:
            param_chain : sampled parameters which are untransformed!
            return_samples : bool, whether to output all samples or just mean and var (see below)
                Note that this can be memory intensive if many samples. If false then doesn't need
                to store each sample.
            diag_cov : whether or not to just compute the diagonal covariance or the full covariance.

        Outputs if return_samples:
            Ysample_means : (n_test, n_samples) mean prediction at the test points for each hyperparmeter sample
                To get the mean to `Ysample_means.mean(axis=1)`
            Ysample_vars : (n_test, n_test, n_samples) prediction covariance at the test points for each hyperparmeter sample
                To change this to a vector of variances at each test point of shape (n_test, n_samples) do `np.diagonal(Ysample_vars).T`
        Outputs if not return_samples:
            Ypred_mean : (n_test, 1) mean of predictions
        """
        assert self.param_chain is not None, "sampling has not yet been done"
        if return_samples:
            if diag_cov:
                compute_var = "diag"
                Ysample_vars = np.zeros((Xnew.shape[0], 1, self.param_chain.shape[0]))
            else:
                compute_var = True
                Ysample_vars = np.zeros((Xnew.shape[0], Xnew.shape[0], self.param_chain.shape[0]))
            Ysample_means = np.zeros((Xnew.shape[0], self.param_chain.shape[0]))
            for i,param in enumerate(self.param_chain):
                self.model.parameters = param
                Ysample_means[:,(i,)], Ysample_vars[:,:,i] = self.model.predict(Xnew, compute_var=compute_var)
            return Ysample_means, Ysample_vars
        else: # just compute the mean
            assert not diag_cov, "covariance not implemented unless returning samples"
            smv = StreamMeanVar(ddof=0)
            for i,param in enumerate(self.param_chain):
                self.model.parameters = param
                smv.include(self.model.predict(Xnew, compute_var=None).squeeze())
            Ypred_mean = smv.mean.reshape((-1,1))
            return Ypred_mean


    def checkgrad(self, decimal=3):
        """ check the gradients of the log likelihood and log prior in the transformed space. This will ensure everything is right: gradients plus transformation """
        step = 1e-6
        params0 = self.params.copy()
        for check in ['likelihood', 'prior']:
            grad_analytic = eval("self.grad_log_%s" % check)
            val0 = eval("self.log_%s" % check)
            grad_exact = np.zeros(grad_analytic.shape)
            for i in range(params0.size):
                tmp = params0.copy()
                tmp[i] += step # take a forward step
                self.params= tmp
                grad_exact[i] = (eval("self.log_%s" % check) - val0) / step

            # protect from nan values incase both analytic and exact are small
            protected_nan = np.logical_and(np.abs(grad_exact) < 1e-8, np.abs(grad_analytic) < 1e-8)

            # now protect against division by zero. I do this by just removing the values that have a small absoute error
            # since if the absolute error is tiny then I don't really care about relative error
            protected_div0 = np.abs(grad_exact-grad_analytic) < 1e-5

            # now artificially change these protected values
            grad_exact[np.logical_or(protected_nan, protected_div0)] = 1.
            grad_analytic[np.logical_or(protected_nan, protected_div0)] = 1.
            try:
                assert_array_almost_equal(grad_exact / grad_analytic, np.ones(grad_exact.shape),
                                          decimal=decimal, err_msg='Gradient ratio did not meet tolerance for log-%s'%check)
            except AssertionError:
                logger.info('Gradient check failed.')
                logger.info('[[Finite-Difference Gradient], [Analytical Gradient]]:\n%s\n' % repr(np.asarray([grad_exact,grad_analytic])))
                raise
            self.params = params0 # reset parameters
        logger.info('Gradient check passed.')


    def optimize_posterior(self, optimizer='lbfgsb', max_iters=1000, messages=True, use_counter=False, factr=None, pgtol=None):
        """
        maximize the log posterior to get MAP estimate

        Inputs:
            optimizer : string
            max_iters : int
                maximum number of optimization iterations
            factr, pgtol : lbfgsb convergence criteria
        """
        logger.debug('Beginning MAP to optimize hyperparameters.')
        #if optimizer is not 'lbfgsb':
            #assert factr is None and pgtol is None, 'these are just for this optimizer (possibly for others but havent checked'
        #optimizer_class = get_optimizer(optimizer) # get the optimizer class based on string given
        #opt = optimizer_class(max_iters=max_iters, gtol=pgtol, bfgs_factor=factr) # get an instance of the object
        #opt.messages = messages # not really sure what this does

        ## now run the optimization
        #x0 = self.params

        ## setup the counter
        #if use_counter:
            #self._counter = solver_counter(disp=True)
        #else:
            #self._counter = None

        ## run the optimization
        #try:
            #opt.run(x0, f_fp=self._MAP_objective_grad, f=None, fp=None)
        #except (KeyboardInterrupt,IndexError): # sometimes interrupting gives index error for scipy sparse matricies it seems
            #logger.info('Keyboard interrupt raised. Cleaning up...')
            #if self._counter is not None and self._counter.backup is not None:# use the backed up copy of parameters from the last iteration
                #self.params = self._counter.backup[1]
                #logger.info('will return best parameter set with log-likelihood = %.4g' % self._counter.backup[0])
        #else:
            #logger.info('Function Evals: %d. Exit status: %s' % (opt.funct_eval,opt.status))
            ## extract the optimal value and set the parameters to this
            #self.params = opt.x_opt # these are the optimal parameters
        #return opt
        raise NotImplementedError("")


    def __str__(self):
        return self.model.__str__()


    def _set_untransformed_parameters(self, parameters):
        """ set parameters that have not be transformed """
        assert parameters.size == self.num_params, "only the free parameters should be passed"
        full_parameters = self.model.parameters
        full_parameters[self.free] = parameters
        self.params = self.model._transform_parameters(full_parameters)[self.free]


    def _MAP_objective_grad(self,transformed_free_parameters):
        """ determines the objective and gradients in the transformed input space """
        try:
            # untransform and internalize parameters
            self.params = transformed_free_parameters
            # compute objective and gradient in untransformed space
            objective = -self.log_p # since we want to minimize
            gradient =  -self.grad_log_p
            # ensure the values are finite
            if not np.isfinite(objective):
                logger.debug('objective is not finite')
            if not np.all(np.isfinite(gradient)):
                logger.debug('some derivatives are non-finite')
        except (LinAlgError, ZeroDivisionError, ValueError):
            logger.error('numerical issue while computing the log-likelihood or gradient.')
            logger.debug('Here is the current model where the failure occured:\n' + self.__str__())
            raise

        # call the counter if ness
        if self._counter is not None:
            msg='log-posterior=%.4g, gradient_norm=%.2g' % (-objective, np.linalg.norm(gradient))
            if self._counter.backup is None or self._counter.backup[0] < -objective: # then update backup
                self._counter(msg=msg,store=(-objective,self.params.copy()))
            else: # don't update backup
                self._counter(msg=msg)
        return objective, gradient


class GPRegression(BaseModel):
    """
    general GP regression model
    """

    def __init__(self, X, Y, kernel, noise_var=1.):
        # call init of the super method
        super(GPRegression, self).__init__()
        # check inputs
        assert X.ndim == 2
        assert Y.ndim == 2
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        assert not np.any(np.isnan(Y))
        self.num_data, self.input_dim = self.X.shape
        if Y.shape[0] != self.num_data:
            raise ValueError('X and Y sizes are inconsistent')
        self.output_dim = self.Y.shape[1]
        if self.output_dim != 1:
            raise RuntimeError('this only deals with 1 response for now')
        assert isinstance(kernel, BaseKernel)
        self.kern = kernel

        # add the noise_var internally
        self.noise_var = np.float64(noise_var)

        # set some defaults
        self.grad_method = 'finite_difference chol'
        return


    def fit(self):
        """ finds the weight vector alpha """
        logger.debug('Fitting; determining weight vector.')
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then need to find the new alpha
            self._alpha = np.linalg.solve(self.kern.cov(x=self.X) + self.noise_var*np.eye(self.num_data),
                                         self.Y)


    def predict(self,Xnew,compute_var=None):
        """
        make predictions at new points

        Inputs:
            Xnew : (M,d) numpy array of points to predict at
            compute_var : whether to compute the variance at the test points
                * None (default) : don't compute variance
                * 'diag' : return the diagonal of the covariance matrix, size (M,1)
                * 'full' : return the full covariance matrix of size (M,M)

        Outputs:
            Yhat : (M,1) numpy array predictions at Xnew
            Yhatvar : only returned if compute_var is not None. See `compute_var` input
                notes for details
        """
        logger.debug('Predicting model at new points.')
        assert Xnew.ndim == 2
        assert Xnew.shape[1] == self.input_dim
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then I need to train
            self.fit()

        # get cross covariance between training and testing points
        Khat = self.kern.cov(x=Xnew, z=self.X)

        # predict the mean at the test points
        Yhat = Khat.dot(self._alpha)

        # predict the variance at the test points
        # TODO: make this more efficient, especially for diagonal predictions
        if compute_var is not None:
            Yhatvar = self.kern.cov(x=Xnew) + self.noise_var*np.eye(Xnew.shape[0]) - \
                    Khat.dot(np.linalg.solve(self.kern.cov(x=self.X) + self.noise_var*np.eye(self.num_data),
                                               Khat.T))
            if compute_var == 'diag':
                Yhatvar = Yhatvar.diag().reshape((-1,1))
            elif compute_var != 'full':
                raise ValueError('Unknown compute_var = %s' % repr(compute_var))
            return Yhat,Yhatvar
        else: # just return the mean
            return Yhat


    def _compute_log_likelihood(self, parameters):
        """
        helper function to compute log likelihood
        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters

        Outputs:
            log_likelihood
        """
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # compute the new covariance
        K = self.kern.cov(self.X)

        # compute the log likelihood
        if 'svd' in self.grad_method: # then compute using svd
            (Q,eig_vals) = np.linalg.svd(K, full_matrices=0, compute_uv=1)[:2]
            log_like = -0.5*np.sum(np.log(eig_vals+self.noise_var)) - \
                        0.5*np.dot(self.Y.T, solve_schur(Q,eig_vals,self.Y,shift=self.noise_var)) - \
                        0.5*self.num_data*np.log(np.pi*2)
        if 'chol' in self.grad_method: # then compute using cholesky factorization
            U = np.linalg.cholesky(K + self.noise_var * np.eye(self.num_data)).T # it returns lower triangular
            log_like = -np.sum(np.log(np.diagonal(U,offset=0,axis1=-1, axis2=-2))) - \
                        0.5*np.dot(self.Y.T, solve_chol(U,self.Y)) - \
                        0.5*self.num_data*np.log(np.pi*2)
        else: # just use logpdf from scipy 
            log_like = mvn.logpdf(self.Y.squeeze(),
                                  np.zeros(self.num_data),
                                  K + self.noise_var * np.eye(self.num_data))
        return log_like


class GPGrief(BaseModel):
    """ GP-GRIEF (GP with GRId-structured Eigen Functions) """

    def __init__(self, X, Y, kern, noise_var=1.):
        """
        GP-GRIEF (GP with GRId-structured Eigen Functions)

        Inputs:
        """
        # call init of the super method
        super(GPGrief, self).__init__()
        # check inputs
        assert X.ndim == 2
        assert Y.ndim == 2
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        assert not np.any(np.isnan(Y))
        self.num_data, self.input_dim = self.X.shape
        if Y.shape[0] != self.num_data:
            raise ValueError('X and Y sizes are inconsistent')
        self.output_dim = self.Y.shape[1]
        if self.output_dim != 1:
            raise RuntimeError('this only deals with 1 response for now')

        # check the kernel
        assert isinstance(kern, GriefKernel)
        assert np.ndim(kern.kern_list) == 1 # This can only be a 1d array of objects
        for i,ki in enumerate(kern.kern_list):
            assert isinstance(ki, BaseKernel) # ensure it is a kernel
            assert ki.n_dims == 1, "currently only 1-dimensional grids allowed"
        self.kern = kern

        # set noise_var internally
        self.noise_var = np.float64(noise_var)

        # add to dependent attributes
        self.dependent_attributes = np.unique(np.concatenate(
            (self.dependent_attributes,
             [
                 '_P', '_Pchol',
                 '_alpha_p', # this is the precomputed W * Phi.T * alpha to speed up predictions
             ])))
        if self.kern.opt_kernel_params: # specify those attributes that change only if the base kernel parameters change
            self.dependent_attributes = np.unique(np.concatenate(
                (self.dependent_attributes,
                 [
                     '_A', '_Phi', # stuff that changes when the base kernel hyperparameters change
                     '_X_last_pred', '_Phi_last_pred', # saved coefficient matrix from the last prediction
                 ])))
        else:# these have to be initialized somewhere
            self._A = None
            self._Phi_last_pred = None

        # set some other default stuff
        if self.kern.opt_kernel_params:
            self.grad_method = 'finite_difference' # TODO: implement adjoint gradients for base kernel hyperparameters!
        else:
            self.grad_method = ['adjoint', 'finite_difference'][0]
        return


    def fit(self, **kwargs):
        """
        finds the weight vector alpha
        """
        # compute using the matrix inversion lemma
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is not None: # check if alpha is already computed
            return
        self._cov_setup()
        self._alpha = self._mv_cov_inv(self.Y)


    def predict(self,Xnew,compute_var=False):
        """
        make predictions at new points

        Inputs:
            Xnew : (M,d) numpy array of points to predict at
            compute_var : whether to compute the variance at the test points

        Outputs:
            Yhat : (M,1) numpy array predictions at Xnew
            Yhatvar : only returned if compute_var is not None. See `compute_var` input
                notes for details
        """
        logger.debug('Predicting model at new points.')
        assert Xnew.ndim == 2
        assert Xnew.shape[1] == self.input_dim
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then I need to train
            self.fit()
        if self._alpha_p is None: # then compute this so future calcs can be done in O(p)
            self._alpha_p = self._Phi.T.dot(self._alpha) * self.kern.w.reshape((-1,1))

        # get cross covariance between training and testing points
        if self._Phi_last_pred is None or not np.array_equal(Xnew, self._X_last_pred): # check if can use previous saved value
            logger.debug("computing Phi at new prediction points")
            self._Phi_last_pred = self.kern.cov(x=Xnew)[0]
            self._X_last_pred = Xnew

        # predict the mean at the test points
        Yhat = self._Phi_last_pred.dot(self._alpha_p)

        # predict the variance at the test points
        if compute_var:
            Yhatvar = self.noise_var*self._Phi_last_pred.dot(cho_solve(self._Pchol, self._Phi_last_pred.T)) + self.noise_var*np.eye(Xnew.shape[0])# see 2.11 of GPML
            return Yhat,Yhatvar
        else: # just return the mean
            return Yhat


    def _cov_setup(self):
        """
        setup the covariance matrix
        """
        if self._P is not None: # then already computed so return
            return
        # get the weights
        self._w = self.kern.w

        # get the p x p matrix A if ness
        if self._A is None: # then compute, note this is expensive
            self._Phi = self.kern.cov(self.X)[0]
            self._A = self._Phi.T.dot(self._Phi) # O(np^2) operation!

        # compute the P matrix and factorize
        self._P = self._A + np.diag(self.noise_var/self._w)
        self._Pchol = cho_factor(self._P)


    def _adjoint_gradient(self,parameters):
        """ compute the log likelihood and the gradient wrt the hyperparameters using the adjoint method """
        assert isinstance(parameters,np.ndarray)

        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]
        gradient = np.zeros(parameters.shape) + np.nan # initialize this

        # compute the log like at the current point. the internal state will be set here
        log_like = self._compute_log_likelihood(parameters)

        # get the gradients wrt the eigenfunction weights (see notebook dec 23 & 28, 2017)
        if self.kern.reweight_eig_funs:
            # compute the data fit gradient
            data_fit_grad = 0.5*np.power(self._Phi.T.dot(self._alpha), 2).squeeze()
            # compute the complexity term gradient
            Pinv_A = cho_solve(self._Pchol, self._A)
            complexity_grad = -0.5*(self._A.diagonal() - (self._A * Pinv_A).sum(axis=0))/(self.noise_var)
            # place the gradients in the vector (it goes last in the list)
            gradient[-self.kern.n_eigs:] = data_fit_grad.squeeze() + complexity_grad.squeeze()
        else:
            Pinv_A = None # specify that this hasn't been computed.

        # get the noise var gradient (see notebook dec 23 & 28, 2017)
        if self.noise_var_constraint != 'fixed':
            if Pinv_A is None: # if eig funs not being reweighted (an edge case), I'll compute this whole term even though i just need the trace
                Pinv_A = cho_solve(self._Pchol, self._A)
            data_fit_grad = 0.5*(self._alpha.T.dot(self._alpha))
            complexity_grad = -0.5*(float(self.num_data) - np.trace(Pinv_A))/self.noise_var
            gradient[0] = data_fit_grad.squeeze() + complexity_grad.squeeze()

        # compute the gradient with respect to the kernel parameters
        if self.kern.opt_kernel_params:
            raise NotImplementedError("adjoint method not implemented for kernel parameter optimiszation, just weights")

        # check to make sure not gradient was missed
        assert not np.any(np.isnan(gradient[free_inds])), "some gradient was missed!"
        return log_like, gradient


    def _compute_log_likelihood(self, parameters):
        """
        compute log likelihood
        """
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # fit the model 
        self.fit()

        # compute the log likelihood
        complexity_penalty = -0.5*self._cov_log_det()
        data_fit = -0.5*self.Y.T.dot(self._alpha)
        log_like = data_fit + complexity_penalty - 0.5*self.num_data*np.log(np.pi*2)
        return log_like


    def _mv_cov(self, x):
        """ matrix vector product with shifted covariance matrix. To get full cov, do `_mv_cov(np.identity(n))` """
        assert x.shape[0] == self.num_data
        assert self._Phi is not None, "cov has not been setup"
        return self._Phi.dot(self._Phi.T.dot(x) * self._w.reshape((-1,1))) + x * self.noise_var


    def _mv_cov_inv(self, x):
        """ matrix vector product with shifted covariance inverse """
        assert x.shape[0] == self.num_data
        assert self._Pchol is not None, "cov has not been setup"
        return (x - self._Phi.dot(cho_solve(self._Pchol, self._Phi.T.dot(x))))/self.noise_var


    def _cov_log_det(self):
        """ compute covariance log determinant """
        assert self._Pchol is not None, "cov has not been setup"
        return 2.*np.sum(np.log(np.diag(self._Pchol[0]))) + np.sum(np.log(self._w)) + float(self.num_data-self.kern.n_eigs)*np.log(self.noise_var)


class GPweb(BaseModel):
    """ GP with weighted basis function kernel (WEB) with O(p^3) computations"""

    def __init__(self, Phi, y, noise_var=1.):
        # call init of the super method
        super(GPweb, self).__init__()

        # initialize counts and stuff
        self.n = y.shape[0]
        y = y.reshape((self.n,1))
        assert Phi.shape[0] == self.n
        self.p= Phi.shape[1]

        # precompute some stuff
        self.r = Phi.T.dot(y)
        self.yTy = y.T.dot(y)
        self.A = Phi.T.dot(Phi)

        # set noise_var internally
        self.noise_var = np.float64(noise_var)

        # initialize the weights
        self.kern = WEBKernel(initial_weights=np.ones(self.p))
        self.grad_method = 'adjoint' #  'finite_difference'

        # add to dependent attributes
        self.dependent_attributes = np.unique(np.concatenate(
            (self.dependent_attributes,
             [
                 '_P', '_Pchol', '_Pinv_r',
                 '_alpha_p', # this is the precomputed W * Phi.T * alpha to speed up predictions
             ])))


    def _compute_log_likelihood(self, parameters):
        # unpack the parameters and initialize some stuff
        self.parameters = parameters # set the internal state

        # precompute some stuff if not already done
        if self._P is None: # compute the P matrix and factorize
            w = self.kern.parameters
            self._P = self.A + np.diag(self.noise_var/w)
            self._Pchol = cho_factor(self._P)
            self._Pinv_r = cho_solve(self._Pchol,self.r)

        # compute the log likelihood
        w = self.kern.parameters
        sig2 = self.noise_var
        datafit = (self.yTy - self.r.T.dot(self._Pinv_r))/sig2
        complexity = 2.*np.sum(np.log(np.diag(self._Pchol[0]))) + np.sum(np.log(w)) + float(self.n-self.p)*np.log(sig2)
        log_likelihood = -0.5*complexity - 0.5*datafit - 0.5*(self.n*np.log(2.*np.pi))
        return log_likelihood.squeeze()


    def _adjoint_gradient(self,parameters):
        """ compute the log likelihood and the gradient wrt the hyperparameters using the adjoint method """
        assert isinstance(parameters,np.ndarray)

        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]
        gradient = np.zeros(parameters.shape) + np.nan # initialize this

        # compute the log like at the current point. the internal state will be set here
        log_like = self._compute_log_likelihood(parameters)

        # precompute terms
        Pinv_A = cho_solve(self._Pchol, self.A)
        w = self.kern.parameters
        sig2 = self.noise_var

        # get the gradients wrt the eigenfunction weights
        data_fit_grad = -np.power((self.r-self.A.dot(self._Pinv_r))/sig2, 2)
        complexity_grad = (self.A.diagonal() - (self.A * Pinv_A).sum(axis=0))/sig2
        # place the gradients in the vector (it goes last in the list)
        gradient[1:] = -0.5*data_fit_grad.squeeze() - 0.5*complexity_grad.squeeze()

        # get the noise var gradient
        data_fit_grad = -(self.yTy-2.*self.r.T.dot(self._Pinv_r) + self._Pinv_r.T.dot(self.A.dot(self._Pinv_r)))/(sig2**2)
        complexity_grad = (float(self.n) - np.trace(Pinv_A))/sig2
        gradient[0] = -0.5*data_fit_grad.squeeze() - 0.5*complexity_grad.squeeze()

        # check to make sure no gradient was missed
        assert not np.any(np.isnan(gradient[free_inds])), "some gradient was missed!"
        return log_like, gradient


    def predict(self,Phi_new,compute_var=False):
        """
        make predictions at new points
        """
        logger.debug('Predicting model at new points.')
        assert Phi_new.ndim == 2
        assert Phi_new.shape[1] == self.p
        parameters = self.parameters # ensure that the internal state is consistent!
        if self._alpha_p is None: # then compute this and save for future use
            if self._Pinv_r is None: # then compute LML to compute this stuff
                self._compute_log_likelihood(parameters);
            w = self.kern.parameters.reshape((-1,1))
            self._alpha_p = (self.r - self.A.dot(self._Pinv_r)) * w / self.noise_var

        # perform the posterior mean computation
        Yhat = Phi_new.dot(self._alpha_p)

        # predict the variance at the test points
        if compute_var == 'diag':
            Yhatvar = self.noise_var*np.sum(Phi_new * cho_solve(self._Pchol, Phi_new.T).T, axis=1, keepdims=True) + self.noise_var
            return Yhat,Yhatvar
        elif compute_var:
            Yhatcov = self.noise_var*Phi_new.dot(cho_solve(self._Pchol, Phi_new.T)) + self.noise_var*np.eye(Phi_new.shape[0])# see 2.11 of GPML
            return Yhat,Yhatcov
        else: # just return the mean
            return Yhat


class GPweb_transformed(BaseModel):
    """ GP with weighted basis function kernel (WEB) with basis functions transformed to give O(p) computations"""

    def __init__(self, Phi, y, noise_var=1.):
        # call init of the super method
        super(GPweb_transformed, self).__init__()

        # initialize counts and stuff
        self.n = y.shape[0]
        y = y.reshape((self.n,1))
        assert Phi.shape[0] == self.n
        self.p_orig = Phi.shape[1]

        # factorize the Phi matrix and determine which transformed bases to keep
        Phit, self.singular_vals, VT = np.linalg.svd(Phi, full_matrices=False, compute_uv=True)
        ikeep = self.singular_vals > 1e-7 # eliminate numerically tiny singular values
        #ikeep = np.cumsum(self.singular_vals)/np.sum(self.singular_vals) < 0.999 # keep 99.9% of variance
        Phit = Phit[:,ikeep]
        self.singular_vals = self.singular_vals[ikeep]
        VT = VT[ikeep,:]
        self.V = VT.T
        self.p = Phit.shape[1]
        if self.p < self.p_orig:
            logger.info("Num Bases decreased from p=%d to p=%d. Only a subspace can now be searched." % (self.p_orig, self.p))

        # precompute some other stuff
        self.PhitT_y = Phit.T.dot(y).squeeze()
        self.PhitT_y_2 = np.power(self.PhitT_y,2)
        self.yTy = np.power(y,2).sum()

        # set noise_var internally
        self.noise_var = np.float64(noise_var)

        # initialize the weights
        self.kern = WEBKernel(initial_weights=np.ones(self.p))
        self.grad_method = 'adjoint' #  'finite_difference'


    def _compute_log_likelihood(self, parameters):
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # compute the log likelihood
        w = self.kern.parameters
        sig2 = self.noise_var
        Pdiag = sig2/w + 1.
        datafit = (self.yTy - np.sum(self.PhitT_y_2/Pdiag))/sig2
        complexity = np.sum(np.log(Pdiag)) + np.sum(np.log(w)) + (self.n-self.p)*np.log(sig2)
        log_likelihood = -0.5*complexity - 0.5*datafit - 0.5*(self.n*np.log(2.*np.pi))
        return log_likelihood


    def _adjoint_gradient(self,parameters):
        """ compute the log likelihood and the gradient wrt the hyperparameters using the adjoint method """
        assert isinstance(parameters,np.ndarray)

        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]
        gradient = np.zeros(parameters.shape) + np.nan # initialize this

        # compute the log like at the current point. the internal state will be set here
        log_like = self._compute_log_likelihood(parameters)

        # get the gradients wrt the eigenfunction weights (see notebook june 1, 2018)
        w = self.kern.parameters
        sig2 = self.noise_var
        # compute the data fit gradient
        data_fit_grad = -self.PhitT_y_2 / np.power(sig2 + w, 2)
        # compute the complexity term gradient
        complexity_grad = 1./(sig2 + w)
        # place the gradients in the vector (it goes last in the list)
        gradient[1:] = -0.5*data_fit_grad.squeeze() - 0.5*complexity_grad.squeeze()

        # get the noise var gradient (see notebook dec 23 & 28, 2017)
        data_fit_grad = (-self.yTy + np.sum(self.PhitT_y_2 * w * (sig2*2. + w) / np.power(sig2 + w, 2))) / sig2**2
        complexity_grad = float(self.n-self.p)/sig2 + np.sum(1./(sig2+w))
        gradient[0] = -0.5*data_fit_grad.squeeze() - 0.5*complexity_grad.squeeze()

        # check to make sure no gradient was missed
        assert not np.any(np.isnan(gradient[free_inds])), "some gradient was missed!"
        return log_like, gradient


    def predict(self,Phi_new,compute_var=False):
        """
        make predictions at new points
        """
        logger.debug('Predicting model at new points.')
        assert Phi_new.ndim == 2
        assert Phi_new.shape[1] == self.p_orig
        self.parameters # ensure that the internal state is consistent!

        # perform the posterior mean computation (see june 4, 2018 notes)
        w = self.kern.parameters
        sig2 = self.noise_var
        Pdiag = sig2/w + 1.
        alpha_p = (self.PhitT_y - self.PhitT_y/Pdiag)  * w / sig2
        Yhat = Phi_new.dot(self.V.dot(alpha_p/self.singular_vals)).reshape((-1,1)) # we use the transformed basis function matrix (see June 1, 2018 notes)

        # predict the variance at the test points
        if compute_var:
            Yhatvar = self.noise_var*Phi_new.dot(Phi_new.T/Pdiag.reshape((-1,1))) + self.noise_var*np.eye(Phi_new.shape[0])# see 2.11 of GPML
            return Yhat,Yhatvar
        else: # just return the mean
            return Yhat


