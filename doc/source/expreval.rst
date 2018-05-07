.. expreval:

Using ``ExprEvaluator``
=======================

This page describes how to use :class:`multilstsq.ExprEvaluator`, which is a tool to parse and `evaluate` (convert to a Python object) expressions provided as strings.

At first, an expression is parsed:

.. code-block:: python

  from multilstsq import ExprEvaluator
  e = ExprEvaluator('a*(2+c*b)')

This expression has 3 `variables`: ``a``, ``b`` and ``c``. We can get this information using:

.. code-block:: python

  e.variables # returns {'a', 'b', 'c'}

As long as these variables are not substituted, it is not possible to evaluate the expression. We can substitute these variables and evaluate the results by doing:

.. code-block:: python

  e(a=3, b=5, c=1) # returns 21, as expected.

However, this is not the typical use case. The goal of :class:`multilstsq.ExprEvaluator` is to enable **partial** substitution, for example substituting ``b`` and ``c``, but not ``a``:

.. code-block:: python

  e2 = e.substitute(constants={'b':5, 'c':1})
  e2(a=3) # Will also return 21, as previously

We can see that ``e2`` has two `constants` (defined variables) and one variable. We can also get the expression as a string for debugging:

.. code-block:: python

  e2.variables # returns {'a'}
  e2.constants # returns {'b', 'c'}
  str(e2)      # returns '(a) * ((2) + ((c) * (b)))'
               # The parentheses are added to ensure correct resolution order

By looking at the expression, we can observe that second operand of the multiplication is fully known, and therefore that it could be `reduced`:

.. code-block:: python

  e3 = e2.reduce()
  str(e3)      # returns '(a) * (__ExprEvaluator_0)'
  e3.variables # returns {'a'}
  e3.constants # returns {'__ExprEvaluator_0'}

We can observe that the expression has indeed be reduced. In the present case, it would have made sense to reduce it to ``(a) * 7``, but instead a (reserved) constant ``__ExprEvaluator_0`` has been created. This makes sense the result of the parenthesis could have been a complex object, like a numpy array.

Note that any python object can be substituted:

.. code-block:: python

  import numpy as np
  e4 = e3.substitute(constants = {'a': np.array([1, 2, 3])})

  e4.eval() # returns array([ 7, 14, 21])
  e4()      # returns the same thing
  e4.reduce().eval() # still the same thing


Complex expression can also be used in the evaluation string. Here's another example:

.. code-block:: python

  import numpy as np
  e = ExprEvaluator('np.linalg.inv(A).dot(y)')
  v1 = np.array([[1, 4], [2, -1]])
  v2 = np.array([9, 0])
  e(A=v1,y=v2) # returns array([1., 2.])

By default, :class:`multilstsq.ExprEvaluator` uses the modules defined at the calling stack frame. This makes ``np`` work in the expression in the example above.
This behavious can be disabled if needed:

.. code-block:: python

  import numpy as np
  e = ExprEvaluator('np.linalg.inv(A).dot(y)',enable_caller_modules=False)
  v1 = np.array([[1, 4], [2, -1]])
  v2 = np.array([9, 0])
  e(A=v1,y=v2) # Will fail, np is not defined

  e(A=v1,y=v2, np=np) # returns array([1., 2.])
s






1. An expression is

.. warning ::

  Using ``ExprEvaluator`` to evaluate expressions from untrusted sources may lead to security vulnerabilities. It should however be safe to use it as long as
  :meth:`multilstsq.ExprEvaluator.eval` is not called.
