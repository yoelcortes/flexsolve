# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:35:01 2019

@author: yoelr
"""
import numpy as np
from . import utils
from .jit_speed import njit_alternative

__all__ = ('false_position', 'bisection', 'bounded_wegstein',
           'bounded_aitken', 'IQ_interpolation', 'find_bracket')

# %% Tools

@njit_alternative
def find_bracket(f, x0, x1, y0=-np.inf, y1=np.inf, yval=0, args=(), maxiter=50):
    """
    Return a bracket within `x0` and `x1` where the objective function, `f`, is 
    certain to have a value of `yval`.
    """
    bisect = utils.bisect
    if y1 < yval:  x1, y1, x0, y0 = x0, y0, x1, y1
    isfinite = np.isfinite
    for iter in range(maxiter):
        x = bisect(x0, x1)
        y = f(x, *args)
        if y > yval:
            x1 = x
            y1 = y
        else:
            x0 = x
            y0 = y
        if isfinite(y0) and isfinite(y1): return (x0, x1, y0, y1)
    raise Exception('failed to find bracket')
    
# %% Solvers

@njit_alternative
def false_position(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-8, ytol=5e-8, args=()):
    """False position solver."""
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < yval:  x1, y1, x0, y0 = x0, y0, x1, y1
    dx = x1 - x0
    df = yval - y0
    abs_ = abs
    err0 = abs_(df)
    err1 = abs_(yval - y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    false_position_iter = utils.false_position_iter
    if x is None or utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx, y0, y1, yval, df, x0)
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    while abs_(dx) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            x1 = x
            y1 = y
        elif y < yval_lb:
            x0 = x
            y0 = y
            df = yval - y0
        else: return x
        dx = x1 - x0
        x = false_position_iter(x0, x1, dx, y0, y1, yval, df, x)
        err = abs_(yval - y)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

@njit_alternative
def bisection(f, x0, x1, y0=None, y1=None, yval=0., xtol=1e-6, ytol=1e-6, args=()):
    """Bisection solver."""
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < yval:  x1, y1, x0, y0 = x0, y0, x1, y1
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    dx = x1 - x0
    abs_ = abs
    err0 = abs_(yval - y0)
    err1 = abs_(yval - y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    bisect = utils.bisect
    x = bisect(x0, x1)
    while abs_(dx) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            x1 = x
        elif y < yval_lb:
            x0 = x
        else: return x
        dx = x1 - x0
        x = bisect(x0, x1)
        err = abs_(yval - y)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

@njit_alternative
def IQ_interpolation(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-8, ytol=5e-8, args=()):
    """Inverse quadratic interpolation solver."""
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < yval:  x1, y1, x0, y0 = x0, y0, x1, y1
    df0 = yval - y0
    dx = x1 - x0
    if x is None or utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx, y0, y1, yval, df0, x0)
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    abs_ = abs
    err0 = abs_(df0)
    err1 = abs_(yval - y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    while abs_(dx) > xtol:
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
        else: return x
        dx = x1 - x0
        x = utils.IQ_iter(y0, y1, y2, yval, x0, x1, x2, dx, df0, x)
        err = abs_(yval - y)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

@njit_alternative
def bounded_wegstein(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-8, ytol=5e-8, args=()):
    """False position solver with Wegstein acceleration."""
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < yval:  x1, y1, x0, y0 = x0, y0, x1, y1
    dx = x1 - x0
    df = yval - y0
    abs_ = abs
    err0 = abs_(df)
    err1 = abs_(yval - y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    if x is None or utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx, y0, y1, yval, df, x0)
    xlast = x
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
    wegstein_iter = utils.scalar_wegstein_iter
    not_within_bounds = utils.not_within_bounds
    while abs_(dx1x0) > xtol:
        y = f(x, *args)
        if y > yval_ub:
            x1 = x
            y1 = y
        elif y < yval_lb:
            x0 = x
            y0 = y
            df = yval - y
        else: return x
        dx1x0 = x1-x0
        g1 = x0 + df*dx1x0/(y1-y0)
        dx = x - xlast
        xlast = x
        x = wegstein_iter(x, dx, g1, g0)
        if not_within_bounds(x, x0, x1): x = g0 = g1
        g1 = g0
        err = abs_(yval - y)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    f(x_best, *args)
    return x

@njit_alternative
def bounded_aitken(f, x0, x1, y0=None, y1=None, x=None, yval=0., xtol=1e-8, ytol=5e-8, args=()):
    """False position solver with Aitken acceleration."""
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < yval:  x1, y1, x0, y0 = x0, y0, x1, y1
    dx1 = x1 - x0
    df = yval - y0
    abs_ = abs
    err0 = abs_(df)
    err1 = abs_(yval - y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    if x is None or utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx1, y0, y1, yval, df, x0)
    yval_ub = yval + ytol
    yval_lb = yval - ytol
    aitken_iter = utils.scalar_aitken_iter
    bisect = utils.bisect
    not_within_bounds = utils.not_within_bounds
    while abs_(dx1) > xtol:
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
        err = abs_(yval - y)
        if err < err_best:
            err_best = err
            x_best = x
        dx0 = x1-x0
        g = x0 + df*dx0/(y1-y0)
        if abs_(dx0) < xtol:
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
        x = aitken_iter(x, gg, dxg, gg - g)
        if not_within_bounds(x, x0, x1): x = bisect(x0, x1)
        err = abs_(yval - y)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

