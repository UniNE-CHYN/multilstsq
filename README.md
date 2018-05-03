MultiLstSq
===============

Least squares is common

This Python 3 module solves:

- solving multiple simultaneous least squares problems
-


This Python 3 module enables to store large structure in a python pickle,
in such a way that the array can be memory-mapped instead of being copied in memory. This module is licensed under the LGPL3 license.

Currently, the container has to be a dictionnary (`mmappickle.dict`), which keys are strings of less than 256 bytes.

It supports any values, but it is only possible to memory-map numpy arrays and numpy masked arrays.

It also supports concurrent access (i.e. you can pass a `mmappickle.dict` as an argument which is called using the `multiprocessing` Python module).

Documentation
=============

Documentation is available at http://multilstsq.readthedocs.io/

Contributing
============

Please post issues and pull requests on github. Alternatively, you can also send your patches by email.

The following tools are used to ensure good code quality:

Tool         | Status
------------ | -------------
travis-ci | [![Build Status](https://travis-ci.org/UniNE-CHYN/multilstsq.svg?branch=master)](https://travis-ci.org/UniNE-CHYN/multilstsq)
AppVeyor | [![Build status](https://ci.appveyor.com/api/projects/status/xxxxx?svg=true)](https://ci.appveyor.com/project/lfasnacht/multilstsq)
Coveralls | [![Coverage Status](https://coveralls.io/repos/github/UniNE-CHYN/multilstsq/badge.svg?branch=master)](https://coveralls.io/github/UniNE-CHYN/multilstsq?branch=master)
