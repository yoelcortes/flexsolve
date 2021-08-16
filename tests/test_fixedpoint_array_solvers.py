# -*- coding: utf-8 -*-
"""
Test iterative methods using a chemical process engineering problem
regarding chemical recycle loops with reactions.

"""
import os
from numba import njit
import numpy as np
import flexsolve as flx
from numpy.testing import assert_allclose
import pytest

conversion = 0.8
stoichiometry = np.array([-1, 0.5, 2, 1, 0.1, 0.001, 0, 0])
feed = np.array([20, 10, 0, 0, 0, 0, 30, 15], dtype=float)
recycle = np.zeros_like(feed)

real_solution = np.array([4.39024390e+00, 1.77804878e+02, 3.51219512e+02, 1.75609756e+02,
                          1.75609756e+01, 1.75609756e-01, 2.70000000e+02, 1.35000000e+02])

real_solution2 = np.array([8.35596757e+01, 1.38220162e+02, 1.92880649e+02, 9.64403243e+01,
                           9.64403244e+00, 9.64403245e-02, 2.70000000e+02, 1.35000000e+02])

@njit(cache=True)
def f(x):
    if (x < 0.).any(): raise Exception('negative values are infeasible')
    recycle = feed.copy()
    recycle[:] = x
    reactor_feed = recycle + feed
    effluent = reactor_feed + (reactor_feed[0] * stoichiometry * conversion)
    product = effluent * 0.1
    return effluent - product

@njit(cache=True)
def f2(x):
    if (x < 0.).any(): raise Exception('negative values are infeasible')
    recycle = feed.copy()
    recycle[:] = x
    reactor_feed = recycle + feed
    reactant = reactor_feed[0]
    rate = reactant * reactant / reactor_feed.sum()
    effluent = reactor_feed + (rate * stoichiometry)
    product = effluent * 0.1
    return effluent - product

def create_plot(p, line=False):
    from matplotlib import pyplot as plt
    xs = np.array(p.xs)
    ys = np.array(p.ys)
    xs = np.abs(xs).sum(0)
    ys = np.abs(ys).sum(0)
    plt.scatter(xs, ys)
    if line:
        plt.plot(xs, ys)

def test_fixedpoint_array_solvers():
    original_feed = feed.copy()
    p = flx.Profiler(f)
    
    solution = flx.wegstein(p, feed, convergenceiter=4, xtol=1e-8, maxiter=200)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution)
    p.archive('Wegstein')
    
    solution = flx.aitken(p, feed, convergenceiter=4, xtol=1e-8, maxiter=200)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution)
    p.archive('Aitken')
    
    solution = flx.fixed_point(p, feed, convergenceiter=4, xtol=5e-8, maxiter=200)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution)
    p.archive('Fixed point')
    
    assert p.sizes() == {'Wegstein': 5, 'Aitken': 5, 'Fixed point': 194}
    
  
def test_fixedpoint_array_solvers2():
    original_feed = feed.copy()
    p = flx.Profiler(f2)
    solution = flx.wegstein(p, feed, xtol=1e-8, maxiter=200)
    assert_allclose(solution, real_solution2)
    p.archive('Wegstein')
    
    with pytest.raises(RuntimeError):
        solution = flx.wegstein(f2, feed, convergenceiter=4, xtol=1e-8, maxiter=200)
    with pytest.raises(RuntimeError):
        solution = flx.wegstein(f2, feed, xtol=1e-8, maxiter=20)
    with pytest.raises(RuntimeError):
        solution = flx.aitken(f2, feed, convergenceiter=4, xtol=1e-8, maxiter=200)
        
    solution = flx.wegstein(p, feed, checkconvergence=False, convergenceiter=4, xtol=1e-8)
    p.archive('Wegstein early termination')
    assert_allclose(solution, real_solution2, rtol=1e-3)
    
    solution = flx.aitken(p, feed, xtol=1e-8, maxiter=10000)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution2)
    p.archive('Aitken')
    
    solution = flx.aitken(p, feed, checkconvergence=False, convergenceiter=4, xtol=1e-8)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution2, rtol=1e-3)
    p.archive('Aitken early termination')
    
    solution = flx.fixed_point(p, feed, xtol=5e-8, maxiter=200)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution2)
    p.archive('Fixed point')
    
    solution = flx.fixed_point(p, feed, maxiter=500, checkconvergence=False, convergenceiter=4, xtol=5e-8)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution2, rtol=1e-3)
    p.archive('Fixed point early termination')
    if os.environ.get("NUMBA_DISABLE_JIT") == '1':
        assert p.sizes() == {'Wegstein': 63, 'Wegstein early termination': 18, 
                             'Aitken': 6801, 'Aitken early termination': 95, 
                             'Fixed point': 191, 'Fixed point early termination': 191}
    else:
        assert p.sizes() == {'Wegstein': 61, 'Wegstein early termination': 18, 
                             'Aitken': 392, 'Aitken early termination': 91,
                             'Fixed point': 191, 'Fixed point early termination': 191}
    
  
def test_conditional_fixedpoint_array_solvers():
    original_feed = feed.copy()
    
    def f_conditional(x):
        y = f(x)
        condition = (np.abs(y - x) >= 5e-8).any()
        return y, condition
    
    p = flx.Profiler(f_conditional)
    
    solution = flx.conditional_wegstein(p, feed)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution)
    p.archive('Wegstein')
    
    solution = flx.conditional_aitken(p, feed)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution)
    p.archive('Aitken')
    
    solution = flx.conditional_fixed_point(p, feed)
    assert_allclose(feed, original_feed)
    assert_allclose(solution, real_solution)
    p.archive('Fixed point')
    
    assert p.sizes() == {'Wegstein': 5, 'Aitken': 5, 'Fixed point': 194}
    