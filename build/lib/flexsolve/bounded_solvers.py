# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:35:01 2019

@author: yoelr
"""
import numpy as np
from .exceptions import SolverError
from collections import namedtuple

__all__ = ('false_position', 'bisection', 'bounded_wegstein',
           'bounded_aitken', 'IQ_interpolation', 'find_bracket')

# %% Tools

Bracket = namedtuple('Braket', ('x0', 'x1', 'y0', 'y1'), module=__name__)        

def get_default_bounds_bounds(f, x0, x1, y0, y1, yval, args):
    if y0 is None:
        y0 = f(x0, *args)
    if y1 is None:
        y1 = f(x1, *args)
    if y1 < yval:
        return x1, y1, x0, y0
    else:
        return x0, y0, x1, y1

def not_within_bounds(x, x0, x1):
    return not (x0 < x < x1 or x1 < x < x0)

def iteration_is_getting_stuck(x, xlast, dx):
    return abs((x - xlast) / dx) < 0.10

def bisect(x0, x1):
    return (x0 + x1) / 2.0

def iter_false_position(x0, x1, dx, y0, y1, yval, df, xlast):
    dy = y1 - y0
    if dy:
        x = x0 + df*dx/dy
        if not_within_bounds(x, x0, x1) or iteration_is_getting_stuck(x, xlast, dx):
            x = bisect(x0, x1)
    else:
        x = bisect(x0, x1)
    return x

def find_bracket(f, x0, x1, y0=None, y1=None, yval=0, args=(), maxiter=50):
    """
    Return a bracket within `x0` and `x1` where the objective function, `f`, is 
    certain to have a value of `yval`.
    """
    np.seterr(divide='raise', invalid='raise')
    for iter in range(maxiter):
        x = bisect(x0, x1)
        y = f(x, *args)
        if y > yval:
            x1 = x
            y1 = y
        else:
            x0 = x
            y0 = y
        if y0 is not None and y1 is not None: 
            return Bracket(x0, x1, y0, y1)
    raise SolverError(f'failed to converge after {maxiter} iterations')
    
def estimate_by_inverse_quadratic_interpolation(y0, y1, y2, yval,
                                                x0, x1, x2, dx, df0, xlast):
    df1 = yval - y1
    df2 = yval - y2
    d01 = df0-df1
    d02 = df0-df2
    d12 = df1-df2
    if all([d12, d02, d01]):
        df0_d12 = df0 / d12
        df1_d02 = df1 / d02
        df2_d01 = df2 / d01
        x = x0*df1_d02*df2_d01 - x1*df0_d12*df2_d01 + x2*df0_d12*df1_d02
        if not_within_bounds(x, x0, x1) or iteration_is_getting_stuck(x, xlast, dx):
            x = bisect(x0, x1)
    else:
        x = iter_false_position(x0, x1, dx, y0, y1, yval, df0, xlast)
    return x

# %% Solvers

def false_position(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-6, ytol=1e-6, args=()):
    """False position solver."""
    np.seterr(divide='raise', invalid='raise')
    _abs = abs
    x0, y0, x1, y1 = get_default_bounds_bounds(f, x0, x1, y0, y1, yval, args)
    dx = x1 - x0
    df = yval - y0
    if x is None or not_within_bounds(x, x0, x1):
        x = iter_false_position(x0, x1, dx, y0, y1, yval, df, x0)
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    while _abs(dx) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            x1 = x
            y1 = y
        elif y < yval_lb:
            x0 = x
            y0 = y
            df = yval - y0
        else: break
        dx = x1 - x0
        x = iter_false_position(x0, x1, dx, y0, y1, yval, df, x)
    return x

def bisection(f, x0, x1, yval=0., xtol=1e-6, ytol=1e-6, args=()):
    """Bisection solver."""
    np.seterr(divide='raise', invalid='raise')
    _abs = abs
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    dx = x1 - x0
    x = bisect(x0, x1)
    while _abs(dx) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            x1 = x
        elif y < yval_lb:
            x0 = x
        else: break
        dx = x1 - x0
        x = bisect(x0, x1)
    return x

def IQ_interpolation(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-6, ytol=1e-6, args=()):
    """Inverse quadratic interpolation solver."""
    np.seterr(divide='raise', invalid='raise')
    _abs = abs
    x0, y0, x1, y1 = get_default_bounds_bounds(f, x0, x1, y0, y1, yval, args)
    df0 = yval - y0
    dx = x1 - x0
    if x is None or not_within_bounds(x, x0, x1):
        x = iter_false_position(x0, x1, dx, y0, y1, yval, df0, x0)
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    while _abs(dx) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            y2 = y1
            x2 = x1
            x1 = x
            y1 = y
        elif y < yval_lb:
            y2 = y0
            x2 = x0
            x0 = x
            y0 = y
            df0 = yval - y
        else: break
        dx = x1 - x0
        x = estimate_by_inverse_quadratic_interpolation(yval, y0, y1, y2,
                                                        x0, x1, x2, dx, df0, x)
    return x

def bounded_wegstein(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-6, ytol=1e-6, args=()):
    """False position solver with Wegstein acceleration."""
    np.seterr(divide='raise', invalid='raise')
    _abs = abs
    x0, y0, x1, y1 = get_default_bounds_bounds(f, x0, x1, y0, y1, yval, args)
    dx = x1 - x0
    df = yval-y0
    if x is None or not_within_bounds(x, x0, x1):
        x = iter_false_position(x0, x1, dx, y0, y1, yval, df, x0)
    x_old = x
    y = f(x, *args)
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    if y > yval_ub:
        x1 = x
        y1 = y
    elif y < yval_lb:
        x0 = x
        y0 = y
        df = yval - y
    else:
        return x
    dx1x0 = x1-x0
    x = g0 = x0 + df*dx1x0/(y1-y0)
    while _abs(dx1x0) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            x1 = x
            y1 = y
        elif y < yval_lb:
            x0 = x
            y0 = y
            df = yval - y
        else: break
        dx1x0 = x1-x0
        g1 = x0 + df*dx1x0/(y1-y0)
        dx = x - x_old
        denominator = dx-g1+g0
        if denominator:
            w = dx / denominator
            x_old = x
            x = w*g1 + (1.-w)*x
            if not_within_bounds(x, x0, x1): x = g0 = g1
            else: g0 = g1                
        else:
            x = g0 = g1
    return x
       
def bounded_aitken(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-6, ytol=1e-6, args=()):
    """False position solver with Aitken acceleration."""
    np.seterr(divide='raise', invalid='raise')
    _abs = abs
    x0, y0, x1, y1 = get_default_bounds_bounds(f, x0, x1, y0, y1, yval, args)
    dx1 = x1-x0
    df = yval-y0
    if x is None or not_within_bounds(x, x0, x1):
        x = iter_false_position(x0, x1, dx1, y0, y1, yval, df, x0)
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    while _abs(dx1) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            x1 = x
            y1 = y
        elif y < yval_lb:
            x0 = x
            y0 = y
            df = yval-y
        else: 
            return x
        dx0 = x1-x0
        g = x0 + df*dx0/(y1-y0)
        if _abs(dx0) < xtol:
            return g
        y = f(g, *args)
        if y > yval_ub:
            x1 = g
            y1 = y
        elif y < yval_lb:
            x0 = g
            y0 = y
            df = yval-y
        else:
            return g
        dx1 = x1-x0
        gg = x0 + df*dx1/(y1-y0)
        dxg = x - g
        denominator = gg + dxg - g
        if denominator:
            x = x - dxg**2. / denominator
        if not_within_bounds(x, x0, x1):
            # Add overshoot to prevent getting stuck
            x = gg + 0.1*(x1+x0-2*gg)*(dx1/dx0)**3. 
    return x
