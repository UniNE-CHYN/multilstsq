import sys

if __name__ == '__main__':
    from mmappickle import mmapdict
    from mmappickle.stubs import EmptyNDArray
    from multilstsq import ExprEvaluator
    import numpy
    import itertools

    d = mmapdict(sys.argv[1])
    d['model'] = 'b0 + b1*x0 + b2*(x0**2)'
    nval = 100000
    shape = (10000, )

    expr = ExprEvaluator(d['model'])

    explanatories = list(sorted(x for x in expr.variables if x.startswith('x')))
    parameters = list(sorted(x for x in expr.variables if x.startswith('b')))

    d['explanatories'] = EmptyNDArray(shape+(nval, len(explanatories)))
    d['response'] = EmptyNDArray(shape+(nval, 1))
    d['parameters'] = numpy.random.normal(size=shape+(len(parameters), ))

    d.vacuum()

    m_explanatories = d['explanatories']
    m_response = d['response']
    m_parameters = d['parameters']


    for pos in itertools.product(*(range(x) for x in shape)):
        print(pos)
        m_explanatories[pos] = numpy.random.normal(size=(nval, len(explanatories)))
        p = dict(zip(parameters, m_parameters[pos]))

        expr_res = expr.substitute(None, p)
        expr_res.enable_call(explanatories)
        m_response[pos] = expr_res(*m_explanatories[pos].T)[:, numpy.newaxis]




    import IPython
    IPython.embed()
