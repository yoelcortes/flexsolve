# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:57:42 2019

@author: yoelr
"""
from .exceptions import SolverError
from . import utils
import numpy as np
from copy import copy

__all__ = ('fixed_point',
           'wegstein', 'conditional_wegstein', 
           'aitken', 'conditional_aitken')

def fixed_point(f, x, xtol=1e-8, args=(), maxiter=50):
    """Iterative fixed-point solver."""
    np.seterr(divide='raise', invalid='raise')
    np_abs = np.abs
    x0 = x
    for iter in range(maxiter):
        x1 = f(x0, *args)
        if (np_abs(x1-x0) < xtol).all(): return x1
        x0 = x1
    raise SolverError(maxiter, x1)

def wegstein(f, x, xtol=1e-8, args=(), maxiter=50):
    """Iterative Wegstein solver."""
    np.seterr(divide='raise', invalid='raise')
    x0 = x
    x1 = g0 = f(x0, *args)
    wegstein_iter = utils.get_wegstein_iter_function(x)
    for iter in range(maxiter):
        dx = x1-x0
        try: g1 = f(x1, *args)
        except:
            x1 = g0
            g1 = f(x1, *args)
        if (np.abs(g1-x1) < xtol).all(): return g1
        x0 = x1
        x1 = wegstein_iter(x1, dx, g1, g0)
        g0 = g1
    raise SolverError(maxiter, g1)

def conditional_wegstein(f, x):
    """Conditional iterative Wegstein solver."""
    np.seterr(divide='raise', invalid='raise')
    x0 = x
    g0, condition = f(x0)
    g1 = x1 = g0
    wegstein_iter = utils.get_wegstein_iter_function(x)
    while condition:
        try: g1, condition = f(x1)
        except:
            x1 = g1
            g1, condition = f(x1)
        dx = x1-x0
        x0 = x1
        x1 = wegstein_iter(x1, dx, g1, g0)
        g0 = g1

def aitken(f, x, xtol=1e-8, args=(), maxiter=50):
    """Iterative Aitken solver."""
    np.seterr(divide='raise', invalid='raise')
    gg = x
    x = copy(x)
    aitken_iter = utils.get_aitken_iter_function(x)
    for iter in range(maxiter):
        try: g = f(x, *args)
        except:
            x = gg.copy()
            g = f(x, *args)
        dxg = x - g
        if (np.abs(dxg) < xtol).all(): return g
        gg = f(g, *args)
        dgg_g = gg - g
        if (np.abs(dgg_g) < xtol).all(): return gg
        x = aitken_iter(x, gg, dxg, dgg_g)
    raise SolverError(maxiter, gg)
    
def conditional_aitken(f, x):
    """Conditional iterative Aitken solver."""
    np.seterr(divide='raise', invalid='raise')
    condition = True
    x = copy(x)
    gg = x
    aitken_iter = utils.get_aitken_iter_function(x)
    while condition:
        try:
            g, condition = f(x)
        except:
            x = gg.copy()
            g, condition = f(x)
        if not condition: return g
        gg, condition = f(g)
        x = aitken_iter(x, gg, x - g, gg - g)