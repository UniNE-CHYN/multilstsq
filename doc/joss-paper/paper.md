---
title: 'multilstsq: Python 3 module to do multiple simultaneous linear regressions'
tags:
  - linear
  - regression
  - least squares
  - lstsq
authors:
 - name: Laurent Fasnacht
   orcid: 0000-0002-9853-8209
   affiliation: 1
affiliations:
 - name: University of Neuchâtel
   index: 1
date: 1 May 2018
bibliography: paper.bib
---

# Summary

Least squares fitting is a underlying method for numerous applications, the most common one being linear regression. It consists in finding the parameters vector ``β°`` which minimizes ``‖ε‖₂`` in the equation ``y = Xβ + ε``, where `X` is the design matrix, `y` the observation vector, and `ε` the error vector.

Since it is a fundamental algorithm, a number of Python 3 implementation exists, with different feature sets and performance, such as:  `numpy.linalg.lstsq`, `scipy.stats.linregress`, `sklearn.linear_model.LinearRegression` and `statsmodel.OLS`.

However, the current available libraries are not designed to work on a large quantity of simultaneous problems, for example solving a least square problem for each pixel of an image. Iterating over a large number of small problems is inefficient. Moreover, when doing linear regression, it is often tedious to build the design matrix `X`.

The goal of `multilstsq` is to work on arrays of problems, with good performance, low memory requirements, reliability and flexibility. It also provides a way to automate the construction of the relevant structures (mostly the design matrix), using a model given as a string. It however does not strive to be a complete statistical library such as what would be provided by `statsmodel` or the language `R`.

To reach these goals, `multilstsq` uses the following techniques:

- It is possible to compute ``β°=(XᵀX)⁻¹Xᵀy`` incrementally, due to the linearity of ``XᵀX`` and ``Xᵀy``, by providing data in chunks.
- Inverting ``XᵀX`` is done by explicit formulas when the dimension is small. This has the advantage of being vector operations which can be applied simultaneously on all problems.
- Masked data are handled as lines of zeros in the design matrix and the observation, which in fact have no effect. This allows adding different amount of data in different subproblems.
- For regression, an expression evaluator is implemented, which converts the input model from the user (for example `b0+b1*x0`) into the complex expression needed to build the design matrix from the vector `X` provided by the user. In that example, it is: `np.concatenate([np.ones(o_dim)[(..., np.newaxis)], ((X)[..., :, 0])[(..., np.newaxis)]])`. This expression evaluator also may be useful for other purposes in other libraries.

As shown in the following figure, this ensures the algorithm has good performance compared to a loop every problem:

![Parallel performance of multilstsq, constant data size.](https://raw.githubusercontent.com/UniNE-CHYN/multilstsq/master/doc/benchmark.png).

# Applications

As linear regression is the underlying operations of many algorithms, many applications can be thought of. The author uses this module for the processing of hyperspectral images, since it is quite common to have to do a regression for each pixel (in the order of 10⁶ to 10⁸ pixels per image).

For example, it can be used to do [flat field correction](https://en.wikipedia.org/wiki/Flat-field_correction), [high dynamic range imaging](https://en.wikipedia.org/wiki/High-dynamic-range_imaging), and image stitching.

# References
