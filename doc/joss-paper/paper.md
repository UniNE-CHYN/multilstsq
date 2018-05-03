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

MultiLstSq is a Python 3 library to do multiple linear regressions simultaneously, in an incremental way, without having to build explicitely the design matrix.

More explicitely, for a given problem ``Y = Xβ + ε``, the goal is to find the parameter vector ``β°`` which minimizes ``‖ε‖₂``. It can be shown that ``β°=(XᵀX)⁻¹XᵀY`` [ref].

For example, if the relation is ``y = β₀ + β₁x₀ + β₂x₀² + β₃x₁``, each row of ``X`` consists in the values ``1 x₀ x₀² x₁``, and each row corresponds to an observation, while ``Y`` is the vector of all responses ``y``. Manually constructing the design matrix ``X`` is error prone, especially during testing with multiple different models, therefore MultiLstSq automates it.

Depending on the number of parameter and the number of observation, ``X`` and ``Y`` can be quite large, making it impractical to directly compute ``β°`` (due to large memory requirements). However, linearity allows us to compute incrementally ``XᵀX`` and ``XᵀY``.

It is quite common to have multiple problems of the same structure (but different ``βᵢ`` values). The classical approach is to do a loop to solve each one individually, but it has performance issues.

This library ensures that the performance doesn't depend on the number of simultaneous regression, but only on the quantity of data, as shown in the following figure:

![Parallel performance of MultiLstSq, constant data size.](https://raw.githubusercontent.com/UniNE-CHYN/mmappickle/master/doc/benchmark.png).

# Applications

As linear regression is the underlying operations of many algorithms, many applications can be thought of. The author uses this module for the processing of hyperspectral images, since it is quite common to have to do a regression for each pixel (in the order of 10⁶ to 10⁸ pixels per image).

For example, it can be used to do [flat field correction](https://en.wikipedia.org/wiki/Flat-field_correction), [high dynamic range imaging](https://en.wikipedia.org/wiki/High-dynamic-range_imaging), and image stitching.

# References
