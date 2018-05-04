import numpy
import itertools
import warnings

MODE_READONLY, MODE_REGRESSION, MODE_VARIANCE = range(3)

class MultiLstSq:
    """
    Multiple least squares problems.

    y = Xβ + ε

    This is a class which can exist in three modes:

    - ``MODE_REGRESSION``: add data to the least square matrices to obtain the mean
    - ``MODE_VARIANCE``: compute the variance
    - ``MODE_READONLY``: the object is frozen.

    The mode can only be switched "forward" (it is not possible to move from ``MODE_VARIANCE`` to ``MODE_REGRESSION`` for example.)

    """
    def __init__(self, problem_dimensions, n_parameters, internal_dtype = numpy.float):
        """
        Create a MultiLstSq object, which is originally in ``MODE_REGRESSION``.

        :param problem_dimensions: Tuple, size of the problem array, it can be ``()`` (0-dimensional), for a single least square problem, or for example ``(800, 600)`` for 800x600 times the regression problem
        :param n_parameters: Number of parameters of the least squares problem.
        :param internal_dtype: :class:`numpy.dtype` data type of the matrices"""

        self._problem_dimensions = problem_dimensions
        self._n_parameters = n_parameters
        self._internal_dtype = internal_dtype

        if type(self._problem_dimensions) != tuple:
            raise TypeError("problem_dimensions should be a tuple")
        if not all(type(x) == int and x>0 for x in self._problem_dimensions):
            raise TypeError("all dimensions in problem_dimensions should be positive integer values")
        if type(self._n_parameters) != int or self._n_parameters <= 0:
            raise TypeError("n_parameters should be a positive integer")

        self._XtX = numpy.zeros(self._problem_dimensions + (self._n_parameters, self._n_parameters), self._internal_dtype)  #(X'X)
        self._Xty = numpy.zeros(self._problem_dimensions + (self._n_parameters, 1), self._internal_dtype)  #X'y

        self._mode = MODE_REGRESSION

        self._rss = 0
        self._n_observations = numpy.zeros(self._problem_dimensions, numpy.int)
        self._cache_beta = None
        self._cache_variance = None

    def switch_to_variance(self):
        """Switch to variance computation mode."""
        if self._mode not in (MODE_REGRESSION, MODE_VARIANCE):
            raise RuntimeError("Only allowed to switch to variance in regression mode.")
        self._mode = MODE_VARIANCE

    def switch_to_read_only(self):
        """Switch to read-only mode."""
        self._mode = MODE_READONLY

    def __validate_dimensions(self, X=None, y=None, w=None):
        """Validate the dimensions of the matrices"""
        assert w is None or y is not None, "Should provide y if providing w"
        assert y is None or X is not None, "Should provide X if providing y"

        if X is not None and (X.ndim != len(self._problem_dimensions) + 2 or X.shape[:-2] != self._problem_dimensions or X.shape[-1] != self._n_parameters):
            X_dim_str = ', '.join([str(d) for d in X.shape])
            dim_str = ', '.join([str(d) for d in self._problem_dimensions]+['<n>',str(self._n_parameters)])

            raise ValueError('Wrong dimensions for X ({} instead of {})'.format(X_dim_str, dim_str))

        if y is not None and (y.shape != self._problem_dimensions + (X.shape[-2], 1)):
            y_dim_str = ', '.join([str(d) for d in y.shape])
            dim_str = ', '.join([str(d) for d in self._problem_dimensions + (X.shape[-2], 1)])

            raise ValueError('Wrong dimensions for y ({} instead of {})'.format(y_dim_str, dim_str))

        if w is not None and (w.shape != y.shape):
            w_dim_str = ', '.join([str(d) for d in w.shape])
            dim_str = ', '.join([str(d) for d in y.shape])
            raise ValueError('Wrong dimensions for w ({} instead of {})'.format(w_dim_str, dim_str))


    def evaluate_at(self, X):
        """Evaluate Xβ"""
        self.__validate_dimensions(X)
        return self._evaluate_at(X)

    def _evaluate_at(self, X):
        return numpy.einsum('...kj,...jl', X, self.beta)

    def add_data(self, X, y, w = None):
        """Add data to the object (depending on the mode, for either mean or variance computation)"""
        self.__validate_dimensions(X, y, w)

        if w is not None:
            assert w.shape[:-2] == self._problem_dimensions
            assert w.shape == y.shape

        return self._add_data(X, y, w)

    def _add_data(self, X, y, w = None):
        if w is not None:
            sweights = numpy.sqrt(w)
            X = X * numpy.repeat(sweights, X.shape[-1], sweights.ndim - 1)
            y = y * sweights
        if self._mode == MODE_REGRESSION:
            self._cache_beta = None
            #Eq to numpy.dot(X.T, X) on last dimensions
            self._XtX += numpy.einsum('...kj,...kl', X, X)
            self._Xty += numpy.einsum('...kj,...kl', X, y)
        elif self._mode == MODE_VARIANCE:
            self._cache_variance = None

            xh = self._evaluate_at(X)
            my_n_obs = numpy.sum(numpy.sum(X != 0, axis = len(self._problem_dimensions) + 1) != 0, axis = len(self._problem_dimensions))

            self._rss += numpy.sum((xh - y) ** 2, axis = len(self._problem_dimensions))[..., 0]
            self._n_observations += my_n_obs #X.shape[-2]
        else:
            raise RuntimeError("Cannot add data when mode=={0}".format(self._mode))

    @property
    def _subproblems_iter(self):
        return itertools.product(*(range(x) for x in self._problem_dimensions))

    def __getstate__(self):
        return {
            'problem_dimensions': self._problem_dimensions,
            'n_parameters': self.n_parameters,
            'internal_dtype': self._internal_dtype,

            #Regression information
            'XtX': self._XtX,
            'Xty': self._Xty,
            'beta': self.beta,

            #Variance information
            'n_observations': self.n_observations,
            'rss': self.rss,
            'variance': self.variance,
        }

    def __setstate__(self, newstate):
        self._problem_dimensions = newstate['problem_dimensions']
        self._n_parameters = newstate['n_parameters']
        self._internal_dtype = newstate.get('internal_dtype', numpy.float)

        self._XtX = newstate['XtX']
        self._Xty = newstate['Xty']
        self._cache_beta = newstate['beta']

        self._n_observations = newstate['n_observations']
        self._rss = newstate['rss']

        self._cache_variance = newstate['variance']

        #Always readonly!
        self._mode = MODE_READONLY

    @property
    def beta(self):
        """The linear coefficients that minimize the least squares criterion."""
        if self._cache_beta is None:
            self._cache_beta = numpy.ma.masked_all(self._Xty.shape, dtype = numpy.float)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #Speedup for simple cases, computed by maxima
                if self._Xty.shape[-2] == 1:
                    #fortran(invert_by_lu(matrix([a[0,0]])).matrix([b[0,0]]));
                    self._cache_beta[...,0,0] = self._Xty[...,0,0]/self._XtX[...,0,0]
                elif self._Xty.shape[-2] == 2:
                    #fortran(invert_by_lu(matrix([a[0,0],a[0,1]],[a[1,0],a[1,1]])).matrix([b[0,0]],[b[1,0]]));
                    self._cache_beta[...,0,0] = self._Xty[...,0,0]*(self._XtX[...,0,1]*self._XtX[...,1,0]/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))+1)/self._XtX[...,0,0]-self._XtX[...,0,1]*self._Xty[...,1,0]/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))
                    self._cache_beta[...,1,0] = self._Xty[...,1,0]/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._Xty[...,0,0]*self._XtX[...,1,0]/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))

                elif self._Xty.shape[-2] == 3:
                    #fortran(invert_by_lu(matrix([a[0,0],a[0,1],a[0,2]],[a[1,0],a[1,1],a[1,2]],[a[2,0],a[2,1],a[2,2]])).matrix([b[0,0]],[b[1,0]],[b[2,0]]))
                    self._cache_beta[...,0,0] = self._Xty[...,0,0]*(-self._XtX[...,0,1]*(-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])-self._XtX[...,1,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*(self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])+1)/self._XtX[...,0,0]+self._Xty[...,1,0]*(self._XtX[...,0,2]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))-self._XtX[...,0,1]*((self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))+1)/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))/self._XtX[...,0,0]+self._Xty[...,2,0]*(self._XtX[...,0,1]*(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))-self._XtX[...,0,2]/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))/self._XtX[...,0,0]
                    self._cache_beta[...,1,0] = self._Xty[...,0,0]*(-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])-self._XtX[...,1,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])+self._Xty[...,1,0]*((self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))+1)/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*self._Xty[...,2,0]/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))
                    self._cache_beta[...,2,0] = self._Xty[...,0,0]*(self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])-self._Xty[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))+self._Xty[...,2,0]/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])

                else:
                    for problem_index in self._subproblems_iter:
                        #This should be, by construction, symmetric
                        #assert(numpy.all(self._XtX == self._XtX.T))
                        #FIXME: check at some other place

                        try:
                            self._cache_beta[problem_index] = numpy.linalg.solve(self._XtX[problem_index], self._Xty[problem_index])
                        except:  #In case of non-solvability
                            pass
        return self._cache_beta

    @property
    def variance(self):
        """Returns the variance/covariance matrix."""
        if self._cache_variance is None:
            self._cache_variance = numpy.ma.masked_all_like(self._XtX)

            d = self._n_observations - self._n_parameters

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._XtX.shape[-2] == 1:
                    #fortran(invert_by_lu(matrix([a[0,0]])));
                    self._cache_variance[..., 0, 0] = 1 / self._XtX[..., 0, 0]
                    self._cache_variance *= (self._rss / d)[..., numpy.newaxis, numpy.newaxis]
                elif self._XtX.shape[-2] == 2:
                    self._cache_variance[..., 0, 0] = (self._XtX[...,0,1]*self._XtX[...,1,0]/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))+1)/self._XtX[...,0,0]
                    self._cache_variance[..., 0, 1] = -self._XtX[...,0,1]/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))
                    self._cache_variance[..., 1, 0] = -self._XtX[...,1,0]/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))
                    self._cache_variance[..., 1, 1] = 1/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])
                    self._cache_variance *= (self._rss / d)[..., numpy.newaxis, numpy.newaxis]
                elif self._XtX.shape[-2] == 3:
                    self._cache_variance[..., 0, 0] = (-self._XtX[...,0,1]*(-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])-self._XtX[...,1,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*(self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])+1)/self._XtX[...,0,0]
                    self._cache_variance[..., 0, 1] = (self._XtX[...,0,2]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))-self._XtX[...,0,1]*((self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))+1)/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))/self._XtX[...,0,0]
                    self._cache_variance[..., 0, 2] = (self._XtX[...,0,1]*(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))-self._XtX[...,0,2]/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))/self._XtX[...,0,0]
                    self._cache_variance[..., 1, 0] = (-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])-self._XtX[...,1,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])
                    self._cache_variance[..., 1, 1] = ((self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))+1)/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])
                    self._cache_variance[..., 1, 2] = -(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))
                    self._cache_variance[..., 2, 0] = (self._XtX[...,1,0]*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,0,0]*(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0]))-self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])
                    self._cache_variance[..., 2, 1] = -(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/((self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0]))
                    self._cache_variance[..., 2, 2] = 1/(self._XtX[...,2,2]-(self._XtX[...,1,2]-self._XtX[...,0,2]*self._XtX[...,1,0]/self._XtX[...,0,0])*(self._XtX[...,2,1]-self._XtX[...,0,1]*self._XtX[...,2,0]/self._XtX[...,0,0])/(self._XtX[...,1,1]-self._XtX[...,0,1]*self._XtX[...,1,0]/self._XtX[...,0,0])-self._XtX[...,0,2]*self._XtX[...,2,0]/self._XtX[...,0,0])

                    self._cache_variance *= (self._rss / d)[..., numpy.newaxis, numpy.newaxis]
                else:
                    for problem_index in self._subproblems_iter:
                        try:
                            self._cache_variance[problem_index] = numpy.linalg.inv(self._XtX[problem_index]) * self._rss[problem_index] / d[problem_index]
                        except:
                            pass

        return self._cache_variance

    @property
    def n_observations(self):
        """Number of observations n."""
        return self._n_observations

    @property
    def rss(self):
        """Residual sum of squares"""
        return self._rss

    @property
    def n_parameters(self):
        """Number of parameters"""
        return self._n_parameters

    @property
    def sigma_2(self):
        """Scaling parameter for the covariance matrix."""
        return self.rss / (self.n_observations - self.n_parameters)


