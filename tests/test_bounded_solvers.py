# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:12:04 2020

@author: yoelr
"""
import flexsolve as flx 
from numpy.testing import assert_allclose

# %% Profile solvers

def test_bounded_solvers():
    x0, x1 = [-5, 5]
    f = lambda x: x**3 - 40 + 2*x 
    x = 3.225240462796626
    assert_allclose(flx.IQ_interpolation(f, x0, x1), x, rtol=1e-5)
    assert_allclose(flx.bisection(f, x0, x1), x, rtol=1e-5)
    assert_allclose(flx.false_position(f, x0, x1), x, rtol=1e-5)
    assert_allclose(flx.find_bracket(f, x0, x1), (2.5, 3.75, -19.375, 20.234375), rtol=1e-5)
    