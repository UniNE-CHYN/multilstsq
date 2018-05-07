Tutorial
========

This page contains some information about how to start if you want to use ``multilstsq`` in your project.

Three different use cases are described in this tutorial, each described in its own section:

1. You want to solve ``y = Xβ`` in the least squares sense, ``y`` and ``X`` are already available.
2. You want to solve ``y = f(x, β)`` in the least squares sense (``β`` is the unknown), where ``f`` is linear in ``β``, but not necessarly in ``x``. This is the linear regression case. Using ``Multiregression``, it is possible to test multiple different ``f``, without changing anything else, so it's great for evaluating multiple models.
3. You only want to use the expression evaluator to add flexibility to some algorithm of your creation. This case is described in its own section: :doc:`expreval`.

We use the following terminology in the following``X`` is the design matrix, ``y`` the observation vector, and `ε` the error vector.
Througout this document, we will call `problem dimension`, the shape of the para
Solving
