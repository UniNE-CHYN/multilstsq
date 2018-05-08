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

It may be required to have variance information. We need to switch to variance mode, and then ...
