# -*- coding: utf-8 -*-
"""
Test iterative methods using a chemical process engineering problem
regarding chemical recycle loops with reactions.

"""
import numpy as np
import flexsolve as flx
from matplotlib import pyplot as plt
from numpy import allclose

conversion = 0.8
stoichiometry = np.array([-1, 0.5, 2, 1, 0.1, 0.001, 0, 0])
feed = np.array([20, 10, 0, 0, 0, 0, 30, 15], dtype=float)
recycle = np.zeros_like(feed)

@flx.njitable
def f(x):
    if (x < 0.).any():
        raise Exception('negative values are infeasible')
    recycle = feed.copy()
    recycle[:] = x
    reactor_feed = recycle + feed
    effluent = reactor_feed + (reactor_feed[0] * stoichiometry * conversion)
    product = effluent * 0.1
    return effluent - product

def create_plot(line=False):
    xs = np.array(p.xs)
    ys = np.array(p.ys)
    xs = np.abs(xs).sum(0)
    ys = np.abs(ys).sum(0)
    plt.scatter(xs, ys)
    if line:
        plt.plot(xs, ys)

original_feed = feed.copy()
p = flx.Profiler(f)
flx.wegstein(p, feed, xtol=1e-8)
assert allclose(feed, original_feed)
p.archive('Wegstein')
flx.aitken(p, feed, xtol=1e-8,maxiter=200)
assert allclose(feed, original_feed)
p.archive('Aitken')
flx.fixed_point_lstsq(p, feed, xtol=5e-8, maxiter=200)
assert allclose(feed, original_feed)
p.archive('Lstsq')
flx.fixed_point(p, feed, xtol=5e-8, maxiter=200)
assert allclose(feed, original_feed)
p.archive('Fixed point')
assert p.sizes() == {'Wegstein': 5, 'Aitken': 5, 'Lstsq': 22, 'Fixed point': 194}

flx.speed_up()
flx.wegstein(f, feed, xtol=5e-8)
flx.aitken(f, feed, xtol=5e-8)