# -*- coding: utf-8 -*-
"""
Test iterative methods using a chemical process engineering problem
regarding chemical recycle loops with reactions.

"""
import numpy as np
import flexsolve as flx
from matplotlib import pyplot as plt

conversion = 0.8
stoichiometry = np.array([-1, 0.5, 2, 1, 0.1, 0.001, 0, 0])
feed = np.array([20, 10, 0, 0, 0, 0, 30, 15], dtype=float)
recycle = np.zeros_like(feed)

def f(x):
    if (x < 0).any(): raise ValueError('values must be possitive')
    recycle[:] = x
    reactor_feed = recycle + feed
    effluent = reactor_feed + (reactor_feed[0] * stoichiometry * conversion)
    product = effluent * 0.1
    recycle[:] = x = effluent - product
    return x

def create_plot(line=False):
    xs = np.array(p.xs)
    ys = np.array(p.ys)
    xs = np.abs(xs).sum(0)
    ys = np.abs(ys).sum(0)
    plt.scatter(xs, ys)
    if line:
        plt.plot(xs, ys)

p = flx.Profiler(f)
solution = flx.fixed_point(p, feed, maxiter=100, lstsq=True)
create_plot(True)
p = flx.Profiler(f)
solution = flx.aitken(p, feed, maxiter=100)
create_plot()
p = flx.Profiler(f)
solution = flx.wegstein(p, feed, maxiter=100)
create_plot()