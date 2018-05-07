import numpy
import sys
import unittest
from multilstsq import MultiLstSq, MultiRegression


class TestMultiLstSq(unittest.TestCase):
    x_i = [1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83]
    y_i = [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46]

    def setUp(self):
        pass

    def _construct_mr_step_by_step(self):
        mr = MultiLstSq((), 2)

        for i in range(2):
            for x, y in zip(self.x_i, self.y_i):
                A = numpy.array([[1, x]])
                b = numpy.array([[y]])
                mr.add_data(A, b)

            mr.switch_to_variance()

        mr.switch_to_read_only()

        with self.assertRaises(RuntimeError):
            mr.add_data(A, b)

        with self.assertRaises(RuntimeError):
            mr.switch_to_variance()

        return mr

    def test_type_validation(self):
        with self.assertRaises(TypeError):
            mr = MultiLstSq([], 1)
        with self.assertRaises(TypeError):
            mr = MultiLstSq((0, 1), 1)
        with self.assertRaises(TypeError):
            mr = MultiLstSq((-1, 1), 1)
        with self.assertRaises(TypeError):
            mr = MultiLstSq((), -1)

        mr = MultiLstSq((), 2)
        with self.assertRaises(ValueError):
            mr.add_data(numpy.array([]), numpy.array([]))
        with self.assertRaises(ValueError):
            mr.add_data(numpy.array([[1, 1]]), numpy.array([]))
        with self.assertRaises(ValueError):
            mr.add_data(numpy.array([[1, 1]]), numpy.array([[1]]), numpy.array([]))

    def test_simple_one_by_one(self):
        mr = self._construct_mr_step_by_step()

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[1, 1], 3.1539, 4)

    def test_simple_all_at_once(self):
        mr = MultiLstSq((), 2)

        A = numpy.hstack((numpy.ones((len(self.x_i), 1)), numpy.array(self.x_i)[:, numpy.newaxis]))
        b = numpy.array(self.y_i)[:, numpy.newaxis]

        mr.add_data(A, b)
        mr.switch_to_variance()
        mr.add_data(A, b)

        mrss = self._construct_mr_step_by_step()

        numpy.testing.assert_array_almost_equal(mr._XtX, mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._Xty, mrss._Xty)

        numpy.testing.assert_array_almost_equal(mr._rss, mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._n_observations, mrss._n_observations)

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[1, 1], 3.1539, 4)

        self.assertAlmostEqual(mr.sigma_2, 0.57619680)

    def test_weights(self):
        mr = MultiLstSq((), 1)
        mr.add_data(numpy.array([[1], [1]]), numpy.array([[0], [3]]), w=numpy.array([[1], [2]]))
        self.assertAlmostEqual(mr.beta[0], 2, 5)

    def test_empty(self):
        mr = MultiLstSq((), 1)
        print(mr.beta)

    def test_1d_all_at_once(self):
        mr = MultiLstSq((3, ), 2)

        A = numpy.hstack((numpy.ones((len(self.x_i), 1)), numpy.array(self.x_i)[:, numpy.newaxis]))
        b = numpy.array(self.y_i)[:, numpy.newaxis]

        A = numpy.array([A] * 3)
        b = numpy.array([b] * 3)

        mr.add_data(A, b)
        mr.switch_to_variance()
        mr.add_data(A, b)

        mrss = self._construct_mr_step_by_step()

        numpy.testing.assert_array_almost_equal(mr._XtX[0], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[1], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[2], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._Xty[0], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[1], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[2], mrss._Xty)

        numpy.testing.assert_array_almost_equal(mr._rss[0], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[1], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._rss[2], mrss._rss)
        numpy.testing.assert_array_almost_equal(mr._n_observations, mrss._n_observations)

        numpy.testing.assert_array_almost_equal(mr.beta[0], mr.beta[1])
        numpy.testing.assert_array_almost_equal(mr.beta[0], mr.beta[2])

        numpy.testing.assert_array_almost_equal(mr.variance[0], mr.variance[1])
        numpy.testing.assert_array_almost_equal(mr.variance[0], mr.variance[2])

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0][0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[0][1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0][0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[0][1, 1], 3.1539, 4)

    def test_2d_all_at_once(self):
        mr = MultiLstSq((2, 2), 2)

        A = numpy.hstack((numpy.ones((len(self.x_i), 1)), numpy.array(self.x_i)[:, numpy.newaxis]))
        b = numpy.array(self.y_i)[:, numpy.newaxis]

        A = numpy.array([[A] * 2] * 2)
        b = numpy.array([[b] * 2] * 2)

        mr.add_data(A, b)
        mr.switch_to_variance()
        mr.add_data(A, b)

        mrss = self._construct_mr_step_by_step()

        numpy.testing.assert_array_almost_equal(mr._XtX[0, 0], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[0, 1], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[1, 0], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[1, 1], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._Xty[0, 0], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[0, 1], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[1, 0], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[1, 1], mrss._Xty)

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

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0][0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[0, 0][1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0][0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[0, 0][1, 1], 3.1539, 4)

    def test_2d_one_by_one(self):
        mr = MultiLstSq((2, 2), 2)

        for i in range(2):
            for x, y in zip(self.x_i, self.y_i):
                A = numpy.array([[[[1, x]]] * 2] * 2)
                b = numpy.array([[[[y]]] * 2] * 2)
                mr.add_data(A, b)

            mr.switch_to_variance()

        mrss = self._construct_mr_step_by_step()

        numpy.testing.assert_array_almost_equal(mr._XtX[0, 0], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[0, 1], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[1, 0], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._XtX[1, 1], mrss._XtX)
        numpy.testing.assert_array_almost_equal(mr._Xty[0, 0], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[0, 1], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[1, 0], mrss._Xty)
        numpy.testing.assert_array_almost_equal(mr._Xty[1, 1], mrss._Xty)

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

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mr.beta[0, 0][0, 0], -39.062, 3)
        self.assertAlmostEqual(mr.beta[0, 0][1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mr.variance[0, 0][0, 0], 8.63185, 5)
        self.assertAlmostEqual(mr.variance[0, 0][1, 1], 3.1539, 4)

    def test_variance(self):
        mr1 = MultiLstSq((), 2)

        for i in range(2):
            for x, y in zip([1, 2, 3], [2.1, 3.9, 6]):
                A = numpy.array([[1, x]])
                b = numpy.array([[y]])
                mr1.add_data(A, b)

            mr1.switch_to_variance()

        mr2 = MultiLstSq((), 2)

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

        # print(mr1.x.T)
        # print(mr2.x.T)

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

    def test_model_multilstsq_simple(self):
        mmr = MultiRegression((), 'b0+b1*x0')

        for i in range(2):
            for x, y in zip(self.x_i, self.y_i):
                A = numpy.array([[x]])
                b = numpy.array([[y]])
                mmr.add_data(A, b)

            mmr.switch_to_variance()

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mmr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mmr.beta[1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mmr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mmr.variance[1, 1], 3.1539, 4)

    def test_model_multilstsq_simple_at_once(self):
        mmr = MultiRegression((), 'b0+b1*x0')

        A = numpy.array(self.x_i)[:, numpy.newaxis]
        b = numpy.array(self.y_i)[:, numpy.newaxis]

        for i in range(2):
            mmr.add_data(A, b)
            mmr.switch_to_variance()

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mmr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mmr.beta[1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mmr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mmr.variance[1, 1], 3.1539, 4)

    def test_model_multilstsq_validation(self):
        with self.assertRaises(ValueError):
            mmr = MultiRegression((), '1')

        mmr = MultiRegression((), 'b0+b1*x0')
        self.assertEqual('b0+b1*x0', mmr.base_model_str)
        self.assertEqual(set(mmr.beta_names), {'b0', 'b1'})

        with self.assertRaises(ValueError):
            mmr.add_data(numpy.array([]), numpy.array([]))
        with self.assertRaises(ValueError):
            mmr.add_data(numpy.array([[1, 1]]), numpy.array([]))
        with self.assertRaises(ValueError):
            mmr.add_data(numpy.array([[1, 1]]), numpy.array([[1]]), numpy.array([]))

    def test_model_multilstsq_simple_masked_at_once(self):
        mmr = MultiRegression((), 'b0+b1*x0')

        Ao = numpy.array(self.x_i)[:, numpy.newaxis]
        bo = numpy.array(self.y_i)[:, numpy.newaxis]

        Am = numpy.ma.masked_all_like(Ao)
        bm = numpy.ma.masked_all_like(bo)

        A = numpy.ma.vstack((Ao, numpy.ma.masked_all((5, 1))))
        b = numpy.ma.vstack((bo, numpy.ma.masked_all((5, 1))))

        for i in range(2):
            mmr.add_data(Am, bo)
            mmr.add_data(Ao, bm)
            mmr.add_data(A, b)
            mmr.switch_to_variance()

        # x is a vertical vector contataining the solution
        self.assertAlmostEqual(mmr.beta[0, 0], -39.062, 3)
        self.assertAlmostEqual(mmr.beta[1, 0], 61.272, 3)
        # variance is the variance-covariance matrix
        self.assertAlmostEqual(mmr.variance[0, 0], 8.63185, 5)
        self.assertAlmostEqual(mmr.variance[1, 1], 3.1539, 4)

    def test_model_multilstsq_simple_at_once_1d(self):
        mmr = MultiRegression((3, ), 'b0+b1*x0')

        A = numpy.array(self.x_i)[:, numpy.newaxis]
        b = numpy.array(self.y_i)[:, numpy.newaxis]

        A = numpy.array([A] * 3)
        b = numpy.array([b] * 3)

        for i in range(2):
            mmr.add_data(A, b)
            mmr.switch_to_variance()

        import pickle

        mmr2 = pickle.loads(pickle.dumps(mmr))

        for i in range(3):
            # x is a vertical vector contataining the solution
            self.assertAlmostEqual(mmr.beta[i][0, 0], -39.062, 3)
            self.assertAlmostEqual(mmr.beta[i][1, 0], 61.272, 3)
            # variance is the variance-covariance matrix
            self.assertAlmostEqual(mmr.variance[i][0, 0], 8.63185, 5)
            self.assertAlmostEqual(mmr.variance[i][1, 1], 3.1539, 4)

            self.assertAlmostEqual(mmr.get_expr_for_idx((i, ))(1), 22.21023062)

            self.assertAlmostEqual(mmr2.get_expr_for_idx((i, ))(1), 22.21023062)

        numpy.testing.assert_almost_equal(
            mmr.apply_expr.substitute(None, {'X': numpy.array([[[1]], [[1]], [[1]]])}).eval(),
            numpy.array([22.21023062, 22.21023062, 22.21023062])
        )

    def test_model_multilstsq_complex(self):
        pmr = MultiRegression((), 'b0+b1*x0+b2*x1')

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

    def test_regression_vs_numpy(self):
        for dim in range(1, 5):
            X = numpy.random.normal(size=(dim * 2, dim))
            y = numpy.random.normal(size=(dim * 2, 1))

            Xmr = numpy.array([numpy.zeros_like(X), X])
            ymr = numpy.array([numpy.zeros_like(y), y])

            mr = MultiLstSq((2, ), dim)
            mr.add_data(Xmr, ymr)
            mr.switch_to_variance()
            mr.add_data(Xmr, ymr)

            beta_hat = mr.beta[1]
            beta_cov = mr.variance

            # FIXME: is there a way to compute the covariance with numpy directly?
            numpy.testing.assert_almost_equal(beta_hat, numpy.linalg.lstsq(X, y, rcond=-1)[0])


if __name__ == '__main__':
    unittest.main()
