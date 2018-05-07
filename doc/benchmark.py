import sys

if __name__ == '__main__':
    from mmappickle import mmapdict
    from multilstsq import MultiRegression
    import numpy
    import time

    d = mmapdict(sys.argv[1], True)

    n = d['explanatories'].shape[1]
    minsh = 1
    maxsh = 200

    times_multilstsq = []
    for s in range(minsh, maxsh):
        mmr = MultiRegression((s, ), d['model'])
        t = time.time()
        mmr.add_data(d['explanatories'][:s, :n // s, :], d['response'][:s, :n // s, :])
        mmr.switch_to_variance()
        mmr.add_data(d['explanatories'][:s, :n // s, :], d['response'][:s, :n // s, :])
        rss = mmr.rss
        beta = mmr.beta
        dt = time.time() - t
        times_multilstsq.append(dt)
        # print(n, s)
        # print(times_multilstsq)

    times_lstsq = []
    for s in range(minsh, maxsh):
        t = time.time()

        beta = numpy.zeros((s, 3))

        for y in range(s):
            explanatories = d['explanatories'][y, :n // s, 0]
            response = d['response'][y, :n // s]

            A = numpy.ones((explanatories.shape[0], 3))
            A[:, 1] = explanatories
            A[:, 2] = explanatories ** 2
            b = response

            beta[y] = numpy.linalg.lstsq(A, b)[0][:, 0]

        dt = time.time() - t
        times_lstsq.append(dt)
        # print(n, s)
        # print(times_lstsq)

    from matplotlib import pyplot as plt

    plt.figure(figsize=(7, 4), dpi=100)
    plt.plot(times_lstsq)
    plt.plot(times_multilstsq)
    plt.xlabel('Number of simultaneous regression')
    plt.ylabel('Time [s] (lower is better)')
    plt.title('Regression time in function of number of simultaneous regression, fixed data size')
    plt.legend(['lstsq', 'multiregression'])
    plt.tight_layout()
    plt.savefig('benchmark.svg')
    plt.savefig('benchmark.png')

    import IPython
    IPython.embed()
