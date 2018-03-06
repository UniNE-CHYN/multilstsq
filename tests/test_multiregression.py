import numpy
import sys
import unittest
from multiregression import Multiregression, ModelMultiregression

class TestMultiregression(unittest.TestCase):
    x_i = [1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83]
    y_i = [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46]
    
    def setUp(self):
        pass
    
    def _construct_mr_step_by_step(self):
        mr = Multiregression((), 2)
        
        for i in range(2):
            for x, y in zip(self.x_i, self.y_i):
                A = numpy.array([[1, x]])
                b = numpy.array([[y]])
                mr.add_data(A, b)
        
            mr.switch_to_variance()
    
        return mr
    
    def test_simple_one_by_one(self):
        mr = self._construct_mr_step_by_step()

        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[1, 1], 3.1539, 4)
        
    def test_simple_all_at_once(self):
        mr = Multiregression((), 2)
        
        A = numpy.hstack((numpy.ones((len(self.x_i), 1)), numpy.array(self.x_i)[:, numpy.newaxis]))
        b = numpy.array(self.y_i)[:, numpy.newaxis]

        mr.add_data(A, b)
        mr.switch_to_variance()
        mr.add_data(A, b)
        
        mrss = self._construct_mr_step_by_step()
        
        numpy.testing.assert_array_almost_equal(mr._AtA, mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._Atb, mrss._Atb)
        
        numpy.testing.assert_array_almost_equal(mr._rss, mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._n_observations, mrss._n_observations)
    
        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[1, 1], 3.1539, 4)
        
    def test_1d_all_at_once(self):
        mr = Multiregression((3, ), 2)
        
        A = numpy.hstack((numpy.ones((len(self.x_i), 1)), numpy.array(self.x_i)[:, numpy.newaxis]))
        b = numpy.array(self.y_i)[:, numpy.newaxis]
        
        A = numpy.array([A] * 3)
        b = numpy.array([b] * 3)
    
        mr.add_data(A, b)
        mr.switch_to_variance()
        mr.add_data(A, b)
    
        mrss = self._construct_mr_step_by_step()
    
        numpy.testing.assert_array_almost_equal(mr._AtA[0], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[1], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[2], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._Atb[0], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[1], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[2], mrss._Atb)
    
        numpy.testing.assert_array_almost_equal(mr._rss[0], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[1], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[2], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._n_observations, mrss._n_observations)
        
        numpy.testing.assert_array_almost_equal(mr.beta[0], mr.beta[1])
        numpy.testing.assert_array_almost_equal(mr.beta[0], mr.beta[2])
        
        numpy.testing.assert_array_almost_equal(mr.variance[0], mr.variance[1])
        numpy.testing.assert_array_almost_equal(mr.variance[0], mr.variance[2])
    
        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0][0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[0][1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0][0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[0][1, 1], 3.1539, 4)
        
    def test_2d_all_at_once(self):
        mr = Multiregression((2, 2), 2)
    
        A = numpy.hstack((numpy.ones((len(self.x_i), 1)), numpy.array(self.x_i)[:, numpy.newaxis]))
        b = numpy.array(self.y_i)[:, numpy.newaxis]
    
        A = numpy.array([[A] * 2] * 2)
        b = numpy.array([[b] * 2] * 2)
    
        mr.add_data(A, b)
        mr.switch_to_variance()
        mr.add_data(A, b)
    
        mrss = self._construct_mr_step_by_step()
    
        numpy.testing.assert_array_almost_equal(mr._AtA[0, 0], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[0, 1], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[1, 0], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[1, 1], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._Atb[0, 0], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[0, 1], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[1, 0], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[1, 1], mrss._Atb)
    
        numpy.testing.assert_array_almost_equal(mr._rss[0, 0], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[0, 1], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[1, 0], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[1, 1], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._n_observations, mrss._n_observations)
    
        numpy.testing.assert_array_almost_equal(mr.beta[0, 0], mr.beta[0, 1])
        numpy.testing.assert_array_almost_equal(mr.beta[0, 0], mr.beta[1, 0])
        numpy.testing.assert_array_almost_equal(mr.beta[0, 0], mr.beta[1, 1])
    
        numpy.testing.assert_array_almost_equal(mr.variance[0, 0], mr.variance[0, 1])
        numpy.testing.assert_array_almost_equal(mr.variance[0, 0], mr.variance[1, 0])
        numpy.testing.assert_array_almost_equal(mr.variance[0, 0], mr.variance[1, 1])
    
        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0][0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[0, 0][1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0][0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[0, 0][1, 1], 3.1539, 4)
        
    def test_2d_one_by_one(self):
        mr = Multiregression((2, 2), 2)
    
        for i in range(2):
            for x, y in zip(self.x_i, self.y_i):
                A = numpy.array([[[[1, x]]] * 2] * 2)
                b = numpy.array([[[[y]]] * 2] * 2)
                mr.add_data(A, b)
        
            mr.switch_to_variance()
    
        mrss = self._construct_mr_step_by_step()
    
        numpy.testing.assert_array_almost_equal(mr._AtA[0, 0], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[0, 1], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[1, 0], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._AtA[1, 1], mrss._AtA)
        numpy.testing.assert_array_almost_equal(mr._Atb[0, 0], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[0, 1], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[1, 0], mrss._Atb)
        numpy.testing.assert_array_almost_equal(mr._Atb[1, 1], mrss._Atb)
    
        numpy.testing.assert_array_almost_equal(mr._rss[0, 0], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[0, 1], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[1, 0], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[1, 1], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._n_observations, mrss._n_observations)
    
        numpy.testing.assert_array_almost_equal(mr.beta[0, 0], mr.beta[0, 1])
        numpy.testing.assert_array_almost_equal(mr.beta[0, 0], mr.beta[1, 0])
        numpy.testing.assert_array_almost_equal(mr.beta[0, 0], mr.beta[1, 1])
    
        numpy.testing.assert_array_almost_equal(mr.variance[0, 0], mr.variance[0, 1])
        numpy.testing.assert_array_almost_equal(mr.variance[0, 0], mr.variance[1, 0])
        numpy.testing.assert_array_almost_equal(mr.variance[0, 0], mr.variance[1, 1])
    
        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0][0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[0, 0][1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0][0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[0, 0][1, 1], 3.1539, 4)
            
    def test_variance(self):
        mr1 = Multiregression((), 2)
    
        for i in range(2):
            for x, y in zip([1, 2, 3], [2.1, 3.9, 6]):
                A = numpy.array([[1, x]])
                b = numpy.array([[y]])
                mr1.add_data(A, b)
    
            mr1.switch_to_variance()
            
        mr2 = Multiregression((), 2)
    
        for i in range(2):
            mr2.add_data(numpy.array([[0, 0]]), numpy.array([[0]]))
            mr2.add_data(numpy.array([[0, 0], [0, 0]]), numpy.array([[0], [0]]))
            
            for x, y in zip([1, 2, 3], [2.1, 3.9, 6]):
                A = numpy.array([[1, x]])
                b = numpy.array([[y]])
                mr2.add_data(A, b)
                
            mr2.add_data(numpy.array([[0, 0]]), numpy.array([[0]]))
            mr2.add_data(numpy.array([[0, 0]]), numpy.array([[0]]))
            mr2.add_data(numpy.array([[0, 0]]), numpy.array([[0]]))
    
            mr2.switch_to_variance()
            
        #print(mr1.x.T)
        #print(mr2.x.T)
        
        numpy.testing.assert_almost_equal(mr1.beta, mr2.beta)
        numpy.testing.assert_almost_equal(mr1.variance, mr2.variance)
        
        return mr1
    
    def test_pickle(self):
        import pickle
        mrss1 = self._construct_mr_step_by_step()
        
        mrss2 = pickle.loads(pickle.dumps(mrss1))
        
        numpy.testing.assert_almost_equal(mrss1.beta, mrss2.beta)
        numpy.testing.assert_almost_equal(mrss1.variance, mrss2.variance)
        
        x = numpy.array([[1, 3.2]])
        numpy.testing.assert_almost_equal(mrss1.evaluate_at(x), mrss2.evaluate_at(x))
        
    def test_model_multiregression_simple(self):
        mmr = ModelMultiregression((), 'b0+b1*x0')
        
        for i in range(2):
            for x, y in zip(self.x_i, self.y_i):
                A = numpy.array([[x]])
                b = numpy.array([[y]])
                mmr.add_data(A, b)
        
            mmr.switch_to_variance()
            
        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mmr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mmr.beta[1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mmr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mmr.variance[1, 1], 3.1539, 4)
        
    def test_model_multiregression_simple_at_once(self):
        mmr = ModelMultiregression((), 'b0+b1*x0')

        A = numpy.array(self.x_i)[:, numpy.newaxis]
        b = numpy.array(self.y_i)[:, numpy.newaxis]
        
        for i in range(2):
            mmr.add_data(A, b)
            mmr.switch_to_variance()

        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mmr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mmr.beta[1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mmr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mmr.variance[1, 1], 3.1539, 4)
        
    def test_model_multiregression_simple_masked_at_once(self):
        mmr = ModelMultiregression((), 'b0+b1*x0')
    
        A = numpy.array(self.x_i)[:, numpy.newaxis]
        b = numpy.array(self.y_i)[:, numpy.newaxis]
        
        A = numpy.ma.vstack((A, numpy.ma.masked_all((5, 1))))
        b = numpy.ma.vstack((b, numpy.ma.masked_all((5, 1))))
    
        for i in range(2):
            mmr.add_data(A, b)
            mmr.switch_to_variance()
    
        #x is a vertical vector contataining the solution
        self.assertAlmostEqual(mmr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mmr.beta[1, 0], 61.272, 3)
        #variance is the variance-covariance matrix
        self.assertAlmostEqual(mmr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mmr.variance[1, 1], 3.1539, 4)
        
    def test_model_multiregression_simple_at_once_1d(self):
        mmr = ModelMultiregression((3, ), 'b0+b1*x0')

        A = numpy.array(self.x_i)[:, numpy.newaxis]
        b = numpy.array(self.y_i)[:, numpy.newaxis]

        A = numpy.array([A] * 3)
        b = numpy.array([b] * 3)

        for i in range(2):
            mmr.add_data(A, b)
            mmr.switch_to_variance()

        for i in range(3):
            #x is a vertical vector contataining the solution
            self.assertAlmostEqual(mmr.beta[i][0, 0], -39.062, 3)
            self.assertAlmostEqual(mmr.beta[i][1, 0], 61.272, 3)
            #variance is the variance-covariance matrix
            self.assertAlmostEqual(mmr.variance[i][0, 0], 8.63185, 5)
            self.assertAlmostEqual(mmr.variance[i][1, 1], 3.1539, 4)
            
    def test_model_multiregression_complex(self):
        pmr = ModelMultiregression((), 'b0+b1*x0+b2*x1')
        
        A = numpy.array([
            [3, 4],
            [5, 1],
            [1, 6],
            [3, 5], 
        ])
        b = numpy.array([
            [12],
            [8],
            [14],
            [14], 
        ])
        pmr.add_data(A, b)
        pmr.switch_to_variance()
        pmr.add_data(A, b)
        
        print(pmr.beta)
        
    def test_ci_interval_modifies_variance_bug(self):
        count=100
        err_sigma=20
        
        x=numpy.sort(numpy.random.uniform(0, 100, size=(count,)))[:,numpy.newaxis]
        y=4+3*x+numpy.random.normal(0, err_sigma, size=(count,1))        

        mmr=ModelMultiregression((), 'b0+b1*x0')
        mmr.add_data(numpy.array(x),numpy.array(y))
        mmr.switch_to_variance()
        mmr.add_data(numpy.array(x),numpy.array(y))        

        var = mmr.variance.copy()
        ci=[mmr.get_confidence_intervals(numpy.array([px]),0.95) for px in x]
        
        numpy.testing.assert_equal(var, mmr.variance)
        
    def test_ci(self):
        mmr = ModelMultiregression((), 'b0+b1*x0')
        
        A = numpy.array(self.x_i)[:, numpy.newaxis]
        b = numpy.array(self.y_i)[:, numpy.newaxis]
    
        for i in range(2):
            mmr.add_data(A, b)
            mmr.switch_to_variance()
            
        print(mmr.get_confidence_intervals(numpy.array([[1.5]]), 0.95))
        print(mmr.get_confidence_intervals(numpy.array([[1.6]]), 0.95))
        
        print(mmr.get_confidence_intervals(numpy.array([[1.5], [1.6]]), 0.95))

if __name__ == '__main__':
    unittest.main()

