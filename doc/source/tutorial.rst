Tutorial
========

This page contains some information about how to start if you want to use ``multilstsq`` in your project.

Three different use cases are described in this documentation:

1. You want to solve ``y = Xβ`` in the least squares sense, ``y`` and ``X`` are already available.
2. You want to solve ``y = f(x, β)`` in the least squares sense (``β`` is the unknown), where ``f`` is linear in ``β``, but not necessarly in ``x``. This is the linear regression case. Using ``Multiregression``, it is possible to test multiple different ``f``, without changing anything else, so it's great for evaluating multiple models.
3. You only want to use the expression evaluator to add flexibility to some algorithm of your creation. This case is described in its own section: :doc:`expreval`.

For the first two cases, we use the following terminology: ``X`` is the `design matrix`, ``y`` the `observation vector`, and `β` the `parameter vector`.

Since ``multilstsq`` is usually used to solve multiple problems simultaneously, we define as `problem dimension` the size of the n-dimensional grid of problems.

The number of problem dimension doesn't have much influence on the way the computation are done, but it makes sense to use dimensions that corresponds to the problem.
For instance, if a characteristics is required for each element of a 20×20 grid, it makes sense to define the problem dimension as ``(20, 20)``, even though it
would be possible to map the problem to a ``(400, )`` or a ``(10, 10, 4)`` problem dimension.

Solving a least squares problem
-------------------------------

First, lets see how to solve a least squares problem.

To keep the example small, let consider ``problem_dimension = (2, 3)``, and the model to be ``y = β₀ + β₁x``.

We have two parameters, and the design matrix consists in rows ``[1 x]``.

Let's see how to do such a regression using first only the class :class:`multilstsq.MultiLstSq`.  This class requires us to build the explicit design matrix, and
does a textbook least square solving on it. First, we need to create the object:

.. code-block:: python

  from multilstsq import MultiLstSq
  import numpy as np

  # First argument is the problem dimensions
  # Second argument is the number of parameters
  mls = MultiLstSq((2, 3), 2)

Now, we need to construct the data to add. We can add one or multiple "observations" at the time. For a single problem, the matrix ``X`` would be of dimensions
``(n_observations, 2)`` (since we have two parameters), while the observation vector should be a column vector of shape: ``(n_observations, 1)``. However, since
we are solving multiple simultaneous least squares problems, we need to add the data for all the problems simultaneously. Therefore in our case the matrix ``X``
is of dimensions ``(2, 3, n_observations, 2)`` (we added the problem dimensions) and the observation vector ``(2, 3, n_observations, 1)``.

.. code-block:: python

  # 4 observations
  X = np.ma.masked_all((2, 3, 4, 2))
  y = np.ma.masked_all((2, 3, 4, 1))

  # This is for β₀
  X[:, :, :, 0] = 1

  # Values of x, as coefficients of β₁
  # The first two dims are the problem index
  # The third is the observation index
  X[0, 0, :, 1] = [1, 2, 3, 4]
  X[0, 1, :, 1] = [1, 2, 1, 2]
  X[0, 2, :, 1] = [1, 1, 1, 1]
  X[1, 0, :, 1] = [-5, -6, 13, 43]
  X[1, 1, :3, 1] = [-1, 0, 1]  # only 3 observations
  X[1, 2, :2, 1] = [4, 8]  # only 2 observations

  # Observations
  y[0, 0, :, 0] = [1, 2, 3, 4]
  y[0, 1, :, 0] = [1.1, 2, 0.9, 2.1]
  y[0, 2, :, 0] = [3, 4, 5, 6]
  y[1, 0, :, 0] = [-5.9, -5.2, 11.9, 42.1]
  y[1, 1, :3, 0] = [1, 2, 3]  # only 3 observations
  y[1, 2, :2, 0] = [4.5, 5]  # only 2 observations

  #Add the data to the regression
  mls.add_data(X, y)

It is possible to run :meth:`multilstsq.MultiLstSq.add_data` multiple times, to "stream" the data to the solver, for example one observation at a time. It may
seem complicated but usually it's quite easy to convert the data matrix into the design matrix. We will see later an even simpler method to do so.

Once we added all the data, we can get ``β``:

.. code-block:: python

  print(mls.beta.shape) # (2, 3, 2, 1)

  # Since it is a big matrix, lets get the value for the first problem
  print(mls.beta[0, 0]) # We get a column vector [0, 1], as expected

  print(mls.beta[0, 2]) # This problem cannot be solved, we get [nan, nan]

It may be required to have variance information. We need to switch to variance mode, and then re-add all the data:

.. code-block:: python

  mls.switch_to_variance()

  # We should add exactly the same data as in the previous step.
  # It doesn't matter if the number of calls is the same.
  mls.add_data(X, y)

Once all the data is added, it is possible to get the covariance matrix for each problem. The covariance matrix is of shape ``(n_parameters, n_parameters)`` for
each subproblem.


.. code-block:: python

  # We can get the covariance matrix
  print(mls.variance.shape)  # (2, 3, 2, 2)

  print(mls.variance[0, 0])  # all zero, as it is a perfect fit

  print(mls.variance[1, 0])  # [[0.299, -0.006], [-0.006, 0.001]]

  print(mls.variance[1, 2])  # all masked: no variance can be computed (only two points for the line)
  print(mls.variance[0, 2])  # all masked: no variance can be computed (not possible to do the fit)


It is also possible to get some additional information, which enables additional processing (like implementing tests):

.. code-block:: python

  print(mls.n_observations)
  # [[4 4 4]
  #  [4 3 2]]

  print(mls.n_parameters)
  # 2

  print(mls.sigma_2)
  # [[0.00 0.01 -- ]
  #  [0.90 0.00 -- ]]

  print(mls.rss)
  # [[0.00 0.03 -- ]
  #  [1.81 0.00 0.0]]


Solving a regression problem
----------------------------

The direct approach shown in the section above works well, but building the design matrix may be impractical when complex model are used, especially if the model
changes. In the following example, for readability we will use ``()`` as the problem size, this means we only do a single regression. Working with a bigger
problem size is exacly similar to the previous section.

Let's say we have the following data:

.. code-block:: python

  import numpy as np
  x0 = np.array([-0.44, -0.52, -0.65, 0.89, -1.15, 1.07, -0.1, 1.05, 1.5, 0.23, -0.87, 1.77, -0.42, 0.43, 1.58, -0.2, -1.69, -1.92, 1.18, -1.18])
  x1 = np.array([1.36, -0.69, -0.96, -0.27, 0.34, -0.02, -0.63, -0.66, 0.96, -0.21, 0.01, -0.06, -1.3, 1.05, 1.08, -1.74, -0.87, 0.72, 0.7, 1.67])
  y = np.array([-0.68, -1.74, -2.24, 2.65, -3.23, 3.29, -0.45, 2.97, 4.96, 0.71, -2.51, 5.35, -1.68, 1.81, 5.25, -1.2, -5.35, -5.41, 3.91, -2.79])

We suspect that this data follows a multilinear relationship according to the model ``y = β₀ + β₁x₀ + β₂x₁``.
The classical approach would be to create a design matrix with rows ``[1, x₀, x₁]``, but we can let the class :class:`multilstsq.MultiRegression` do the work for us:

.. code-block:: python

  from multilstsq import MultiRegression
  mr = MultiRegression((), 'b0 + b1 * x0 + b2 * x1')

  #Note that X is a matrix of two columns x₀, x₁
  X = np.array([x0, x1]).T
  Y = np.array([y]).T

  mr.add_data(X,Y)
  mr.switch_to_variance()
  mr.add_data(X,Y)

  mr.rss  # 0.002193175857390197

  # etc.

Now, let say we suspect that the model is not what we expected, we can compare models easily:

.. code-block:: python

  models = [
    'b0 + b1 * x0 + b2 * x1',
    'b0 + b1 * x0 + b2 * x1 + b3 * (x1**2)',
    'b0 + b1 * x0 + b2 * x1 + b3 * (x0**2)',
    'b0 + b1 * x0 + b2 * x1 + b3 * (x0**2) + b4 * (x1**2)',
  ]

  for model in models:
    mr = MultiRegression((), model)
    mr.add_data(X,Y)
    mr.switch_to_variance()
    mr.add_data(X,Y)

    print('{:0.05f}'.format(mr.rss), model)

  # We obtain the following output:
  # 0.00278 b0 + b1 * x0 + b2 * x1
  # 0.00270 b0 + b1 * x0 + b2 * x1 + b3 * (x1**2)
  # 0.00016 b0 + b1 * x0 + b2 * x1 + b3 * (x0**2)
  # 0.00014 b0 + b1 * x0 + b2 * x1 + b3 * (x0**2) + b4 * (x1**2)

We can see that the best model is likely to be ``y = β₀ + β₁x₀ + β₂x₁ + β₂x₀²``. Using this kind of technique it is very simple to make step-wise regression.

The model string can be any valid Python expression, but requires it to be linear in ``b``'s. Each variable ``b0``, ``b1``, ``b2``... corresponds to a parameter,
while ``x0``, ``x1``, ``x2``... corresponds to columns of the matrix ``X``.

