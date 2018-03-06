import numpy
import itertools
import scipy, scipy.linalg, scipy.special, scipy.sparse

from .expreval import ExprEvaluator, RegrExprEvaluator

MODE_READONLY, MODE_REGRESSION, MODE_VARIANCE = range(3)

class Multiregression:
    def __init__(self, problem_dimensions, n_parameters, internal_dtype = numpy.float):
        """Problem dimensions is the size of the problem, it can be () (0-dimentional), or for example (800,600) for 800x600 times the regression problem"""
        self._problem_dimensions = problem_dimensions
        self._n_parameters = n_parameters
        self._internal_dtype = internal_dtype
        
        assert type(self._problem_dimensions) == tuple
        assert all(type(x) == int for x in self._problem_dimensions)
        assert type(self._n_parameters) == int
        
        self._AtA = numpy.zeros(self._problem_dimensions + (self._n_parameters, self._n_parameters), self._internal_dtype)  #(A'A)
        self._Atb = numpy.zeros(self._problem_dimensions + (self._n_parameters, 1), self._internal_dtype)  #A'b
        
        self._mode = MODE_REGRESSION
        
        self._rss = 0
        self._n_observations = numpy.zeros(self._problem_dimensions, numpy.int)
        self._cache_beta = None
        self._cache_variance = None
        
    def switch_to_variance(self):
        assert self._mode in (MODE_REGRESSION, MODE_VARIANCE), "Only allowed to switch to variance in regression mode."
        self._mode = MODE_VARIANCE
        
    def switch_to_read_only(self):
        self._mode = MODE_READONLY
        
    def evaluate_at(self, A):
        """Evaluate the lhs at A"""
        assert A.ndim == len(self._problem_dimensions) + 2
        assert A.shape[:-2] == self._problem_dimensions
        assert A.shape[-1] == self._n_parameters
        return self._evaluate_at(A)
        
    def _evaluate_at(self, A):
        return numpy.einsum('...kj,...jl', A, self.beta)
        
    def add_data(self, A, b, weights = None):
        assert A.ndim == len(self._problem_dimensions) + 2
        assert A.shape[:-2] == self._problem_dimensions
        assert A.shape[-1] == self._n_parameters, "{0}, {1}".format(A.shape, self._n_parameters)
        assert b.ndim == len(self._problem_dimensions) + 2
        assert b.shape == self._problem_dimensions + (A.shape[-2], 1)
        
        if weights is not None:
            assert weights.shape[:-2] == self._problem_dimensions
            assert weights.shape == b.shape
        
        return self._add_data(A, b, weights)
    
    def _add_data(self, A, b, weights = None):
        if weights is not None:
            sweights = numpy.sqrt(weights)
            A = A * numpy.repeat(sweights, A.shape[-1], sweights.ndim - 1)
            b = b * sweights
        if self._mode == MODE_REGRESSION:
            self._cache_beta = None
            #Eq to numpy.dot(A.T, A) on last dimensions
            self._AtA += numpy.einsum('...kj,...kl', A, A)
            self._Atb += numpy.einsum('...kj,...kl', A, b)
        elif self._mode == MODE_VARIANCE:
            self._cache_variance = None
            if self.beta is None:
                return
            
            xh = self._evaluate_at(A)
            my_n_obs = numpy.sum(numpy.sum(A != 0, axis = len(self._problem_dimensions) + 1) != 0, axis = len(self._problem_dimensions))
            
            self._rss += numpy.sum((xh - b) ** 2, axis = len(self._problem_dimensions))[..., 0]
            self._n_observations += my_n_obs #A.shape[-2]
        else:
            raise ValueError("Cannot add data when mode=={0}".format(self._mode))
    
    @property
    def _subproblems_iter(self):
        return itertools.product(*(range(x) for x in self._problem_dimensions))
    
    def __getstate__(self):
        return {
            'problem_dimensions': self._problem_dimensions,
            'n_parameters': self.n_parameters,
            'internal_dtype': self._internal_dtype,
            
            #Regression information
            'AtA': self._AtA,
            'Atb': self._Atb,            
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
        
        self._AtA = newstate['AtA']
        self._Atb = newstate['Atb']
        self._cache_beta = newstate['beta']
        
        self._n_observations = newstate['n_observations']
        self._rss = newstate['rss']
        
        self._cache_variance = newstate['variance']
        
        #Always readonly!
        self._mode = MODE_READONLY
    
    @property
    def beta(self):
        if self._cache_beta is None:
            self._cache_beta = numpy.ma.masked_all(self._Atb.shape, dtype = numpy.float)
            
            #Speedup for simple cases, computed by maxima
            if self._Atb.shape[-2] == 1:
                #fortran(invert_by_lu(matrix([a[0,0]])).matrix([b[0,0]]));
                self._cache_beta[...,0,0] = self._Atb[...,0,0]/self._AtA[...,0,0]
            elif self._Atb.shape[-2] == 2:
                #fortran(invert_by_lu(matrix([a[0,0],a[0,1]],[a[1,0],a[1,1]])).matrix([b[0,0]],[b[1,0]]));
                self._cache_beta[...,0,0] = self._Atb[...,0,0]*(self._AtA[...,0,1]*self._AtA[...,1,0]/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))+1)/self._AtA[...,0,0]-self._AtA[...,0,1]*self._Atb[...,1,0]/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))
                self._cache_beta[...,1,0] = self._Atb[...,1,0]/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._Atb[...,0,0]*self._AtA[...,1,0]/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))

            elif self._Atb.shape[-2] == 3:
                #fortran(invert_by_lu(matrix([a[0,0],a[0,1],a[0,2]],[a[1,0],a[1,1],a[1,2]],[a[2,0],a[2,1],a[2,2]])).matrix([b[0,0]],[b[1,0]],[b[2,0]]))
                self._cache_beta[...,0,0] = self._Atb[...,0,0]*(-self._AtA[...,0,1]*(-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])-self._AtA[...,1,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*(self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])+1)/self._AtA[...,0,0]+self._Atb[...,1,0]*(self._AtA[...,0,2]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))-self._AtA[...,0,1]*((self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))+1)/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))/self._AtA[...,0,0]+self._Atb[...,2,0]*(self._AtA[...,0,1]*(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))-self._AtA[...,0,2]/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))/self._AtA[...,0,0]
                self._cache_beta[...,1,0] = self._Atb[...,0,0]*(-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])-self._AtA[...,1,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])+self._Atb[...,1,0]*((self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))+1)/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*self._Atb[...,2,0]/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))
                self._cache_beta[...,2,0] = self._Atb[...,0,0]*(self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])-self._Atb[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))+self._Atb[...,2,0]/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])
                
            else:
                for problem_index in self._subproblems_iter:
                    #This should be, by construction, symmetric
                    #assert(numpy.all(self._AtA == self._AtA.T))
                    #FIXME: check at some other place
                    
                    try:
                        self._cache_beta[problem_index] = numpy.linalg.solve(self._AtA[problem_index], self._Atb[problem_index])
                    except:
                        pass
        return self._cache_beta
    
    @property
    def variance(self):
        if self._cache_variance is None:
            self._cache_variance = numpy.ma.masked_all_like(self._AtA)
            
            d = self._n_observations - self._n_parameters
            
            if self._AtA.shape[-2] == 1:
                #fortran(invert_by_lu(matrix([a[0,0]])));
                self._cache_variance[..., 0, 0] = 1 / self._AtA[..., 0, 0]
                self._cache_variance *= (self._rss / d)[..., numpy.newaxis, numpy.newaxis]
            elif self._AtA.shape[-2] == 2:
                self._cache_variance[..., 0, 0] = (self._AtA[...,0,1]*self._AtA[...,1,0]/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))+1)/self._AtA[...,0,0]
                self._cache_variance[..., 0, 1] = -self._AtA[...,0,1]/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))
                self._cache_variance[..., 1, 0] = -self._AtA[...,1,0]/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))
                self._cache_variance[..., 1, 1] = 1/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])
                self._cache_variance *= (self._rss / d)[..., numpy.newaxis, numpy.newaxis]
            elif self._AtA.shape[-2] == 3:
                self._cache_variance[..., 0, 0] = (-self._AtA[...,0,1]*(-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])-self._AtA[...,1,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*(self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])+1)/self._AtA[...,0,0]
                self._cache_variance[..., 0, 1] = (self._AtA[...,0,2]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))-self._AtA[...,0,1]*((self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))+1)/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))/self._AtA[...,0,0],
                self._cache_variance[..., 0, 2] = (self._AtA[...,0,1]*(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))-self._AtA[...,0,2]/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))/self._AtA[...,0,0]
                self._cache_variance[..., 1, 0] = (-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])-self._AtA[...,1,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])
                self._cache_variance[..., 1, 1] = ((self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))+1)/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])
                self._cache_variance[..., 1, 2] = -(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))
                self._cache_variance[..., 2, 0] = (self._AtA[...,1,0]*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,0,0]*(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0]))-self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])
                self._cache_variance[..., 2, 1] = -(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/((self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0]))
                self._cache_variance[..., 2, 2] = 1/(self._AtA[...,2,2]-(self._AtA[...,1,2]-self._AtA[...,0,2]*self._AtA[...,1,0]/self._AtA[...,0,0])*(self._AtA[...,2,1]-self._AtA[...,0,1]*self._AtA[...,2,0]/self._AtA[...,0,0])/(self._AtA[...,1,1]-self._AtA[...,0,1]*self._AtA[...,1,0]/self._AtA[...,0,0])-self._AtA[...,0,2]*self._AtA[...,2,0]/self._AtA[...,0,0])
                
                self._cache_variance *= (self._rss / d)[..., numpy.newaxis, numpy.newaxis]
            else:
                for problem_index in self._subproblems_iter:
                    try:
                        self._cache_variance[problem_index] = numpy.linalg.inv(self._AtA[problem_index]) * self._rss[problem_index] / d[problem_index]
                    except:
                        pass
        
        return self._cache_variance
    
    @property
    def n_observations(self):
        return self._n_observations
    
    @property
    def rss(self):
        return self._rss
    
    @property
    def n_parameters(self):
        return self._n_parameters
    
    @property
    def sigma_2(self):
        return self.rss / (self.n_observations - self.n_parameters)
    
    def get_confidence_intervals_old(self, X, pvalue):
        assert X.ndim == len(self._problem_dimensions) + 2
        assert X.shape[:-2] == self._problem_dimensions
        assert X.shape[-1] == self._n_parameters

        my_qf = scipy.special.fdtri(self.n_parameters, self.n_observations - self.n_parameters, pvalue)
        
        #FIXME: find a way to do this in one step
        ret = numpy.ma.masked_all(self._problem_dimensions + (X.shape[-2], 1))
        for problem_index in self._subproblems_iter:
            ret[problem_index] = numpy.sqrt(numpy.einsum('...i,...ij,...j->...', X[problem_index], self.variance[problem_index], X[problem_index]) * self.n_parameters * my_qf[problem_index])[:, numpy.newaxis]
        #return numpy.sqrt(numpy.einsum('...i,...ij,...j->...', X, self.variance, X) * self.n_parameters * my_qf)
        return ret[..., 0, 0]
    
    def get_confidence_intervals(self, X, pvalue):
        assert X.ndim == len(self._problem_dimensions) + 2
        assert X.shape[:-2] == self._problem_dimensions
        assert X.shape[-1] == self._n_parameters

        #q = scipy.special.ndtri(1 - (1 - pvalue) / 2)
        q = scipy.special.stdtrit((self._n_observations - self._n_parameters), (1 - (1 - pvalue) / 2))

        #FIXME: find a way to do this in one step
        ret = numpy.ma.masked_all(self._problem_dimensions + (X.shape[-2], 1))
        for problem_index in self._subproblems_iter:
            Qxx = self.variance[problem_index] / self._rss[problem_index] * (self._n_observations[problem_index] - self._n_parameters)
            Vph = self.sigma_2[problem_index] * numpy.einsum('...i,...ij,...j->...', X[problem_index], Qxx, X[problem_index])
            ret[problem_index] = q[problem_index] * numpy.sqrt(Vph)[:, numpy.newaxis]
            
        return ret
    
    def get_prediction_intervals(self, X, pvalue):
        assert X.ndim == len(self._problem_dimensions) + 2
        assert X.shape[:-2] == self._problem_dimensions
        assert X.shape[-1] == self._n_parameters

        #q = scipy.special.ndtri(1 - (1 - pvalue) / 2)
        if pvalue is None:
            #We only want to get the multiplicative factor
            q = numpy.ones(self._problem_dimensions)
        else:
            #Use student-t quantiles
            q = scipy.special.stdtrit((self._n_observations - self._n_parameters), (1 - (1 - pvalue) / 2))

        assert type(self.variance) != int
        assert type(self._rss) != int
        assert type(self._n_observations) != int

        #FIXME: find a way to do this in one step
        ret = numpy.ma.masked_all(self._problem_dimensions + (X.shape[-2], 1))
        for problem_index in self._subproblems_iter:
            Qxx = self.variance[problem_index] / self._rss[problem_index] * (self._n_observations[problem_index] - self._n_parameters)
            Vph = self.sigma_2[problem_index] * numpy.einsum('...i,...ij,...j->...', X[problem_index], Qxx, X[problem_index])
            ret[problem_index] = q[problem_index] * numpy.sqrt(self.sigma_2[problem_index] + Vph)[:, numpy.newaxis]

        return ret
        
class ModelMultiregression(Multiregression):
    def __init__(self, problem_dimensions, model_str, internal_dtype = numpy.float):
        self._build_expressions(problem_dimensions, model_str)
        Multiregression.__init__(self, problem_dimensions, len(self._base_model.parameter_variables), internal_dtype)
        
    def _build_expressions(self, problem_dimensions, model_str):
        self._base_model_str = model_str
        self._base_model = RegrExprEvaluator(model_str)
        if len(self._base_model.explanatory_variables) == 0:
            self._n_explanatory_min = 0
        else:
            self._n_explanatory_min = max(int(x[1:]) for x in self._base_model.explanatory_variables) + 1
        
        if len(self._base_model.parameter_variables) > 0:
            self._conversion_expression = ExprEvaluator(
                'numpy.concatenate([{0}], axis={1})'.format(','.join('(numpy.ones(o_dim)*({0}))[...,numpy.newaxis]'.format(var) for var in self._base_model.parameter_variables), len(problem_dimensions) + 1)
            )
        else:
            self._conversion_expression = ExprEvaluator(
                'numpy.empty({0}+(A.shape[-2],0))'.format(problem_dimensions)
            )
        self._conversion_expression = self._conversion_expression.substitute(
            dict((var, self._base_model.find_coeff_for(var)) for var in self._base_model.parameter_variables)
        )
        self._conversion_expression = self._conversion_expression.substitute(
            dict((var, 'A[..., :, {0}]'.format(var[1:])) for var in self._base_model.explanatory_variables),
        )
    
        self._apply_expression = self._base_model.substitute(
            dict((var, "beta[...,{0},:]".format(idx)) for idx, var in enumerate(self._base_model.parameter_variables))
        ).substitute(
            dict((var, 'A[..., :, {0}]'.format(var[1:])) for var in self._base_model.explanatory_variables),
        )
        
        self._apply_expression = ExprEvaluator('x[...,0]').substitute({'x': self._apply_expression,})
        
    @property
    def base_model_str(self):
        return self._base_model_str
    
    @property
    def beta_names(self):
        return self._base_model.parameter_variables
        
    def add_data(self, A, b, weights = None):
        assert A.ndim == len(self._problem_dimensions) + 2
        assert A.shape[:-2] == self._problem_dimensions
        assert A.shape[-1] >= self._n_explanatory_min
        assert b.ndim == len(self._problem_dimensions) + 2
        assert b.shape == self._problem_dimensions + (A.shape[-2], 1)
        
        if weights is not None:
            assert weights.shape == b.shape
        
        new_A = self._conversion_expression.substitute(None, {'o_dim': A.shape[:-1], 'A': A,}).eval()
        
        maskA = None
        maskb = None
        if isinstance(A, numpy.ma.MaskedArray) and A.mask.shape == A.shape:
            maskA = numpy.any(A.mask, axis = len(self._problem_dimensions) + 1)
            new_A = new_A.data.copy()
            new_A[maskA] = 0
        if isinstance(b, numpy.ma.MaskedArray) and b.mask.shape == b.shape:
            maskb = numpy.any(b.mask, axis = len(self._problem_dimensions) + 1)
            b = b.data.copy()
            b[maskb] = 0
            
        if maskA is not None or maskb is not None:
            #generate valid
            if maskA is not None and maskb is not None:
                valid = ~numpy.logical_or(maskA, maskb)
            elif maskA is not None:
                valid = ~maskA
            elif maskb is not None:
                valid = ~maskb
        
            validA = numpy.repeat(valid[..., :, numpy.newaxis], new_A.shape[-1], len(self._problem_dimensions) + 1)
            validb = valid[..., :, numpy.newaxis]
            
            return super().add_data(new_A * validA, b * validb, weights)
        else:
            return super().add_data(new_A, b, weights)
    
    def get_confidence_intervals(self, X, pvalue):
        assert X.ndim == len(self._problem_dimensions) + 2
        assert X.shape[:-2] == self._problem_dimensions
        assert X.shape[-1] >= self._n_explanatory_min
        
        new_X = self._conversion_expression.substitute(None, {'o_dim': X.shape[:-1], 'A': X,}).eval()
        return super().get_confidence_intervals(new_X, pvalue)
    
    def get_prediction_intervals(self, X, pvalue):
        assert X.ndim == len(self._problem_dimensions) + 2
        assert X.shape[:-2] == self._problem_dimensions
        assert X.shape[-1] >= self._n_explanatory_min

        new_X = self._conversion_expression.substitute(None, {'o_dim': X.shape[:-1], 'A': X,}).eval()
        return super().get_prediction_intervals(new_X, pvalue)
        
    
    def get_expr_for_idx(self, pb_idx):
        assert type(pb_idx) == tuple
        expr = self._base_model.substitute(None, 
            dict((var, self.beta[pb_idx+(varidx,0)]) for varidx, var in enumerate(self._base_model.parameter_variables))
        ).reduce()
        expr.enable_call(['x{0}'.format(x) for x in range(self._n_explanatory_min + 1)])
        return expr
        
    
    @property
    def apply_expr(self):
        return self._apply_expression.substitute(None, {'beta':self.beta})
        
    def __getstate__(self):
        d = super().__getstate__()
        d['base_model_str'] = self._base_model_str
        return d

    def __setstate__(self, newstate):    
        self._build_expressions(newstate['problem_dimensions'], newstate['base_model_str'])
        return super().__setstate__(newstate)
        
