# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:35:01 2019

@author: yoelr
"""
import numpy as np
from .jit_speed import njit_alternative
from . import utils
__all__ = ('false_position', 'bisection', 
           'IQ_interpolation', 'find_bracket')

# %% Tools

@njit_alternative
def find_bracket(f, x0, x1, y0=-np.inf, y1=np.inf, args=(), maxiter=50):
    """
    Return a bracket within `x0` and `x1` where the objective function, `f`, is 
    certain to have a root.
    """
    bisect = utils.bisect
    if y1 < 0.:  x1, y1, x0, y0 = x0, y0, x1, y1
    isfinite = np.isfinite
    for iter in range(maxiter):
        x = bisect(x0, x1)
        y = f(x, *args)
        if y > 0.:
            x1 = x
            y1 = y
        else:
            x0 = x
            y0 = y
        if isfinite(y0) and isfinite(y1): return (x0, x1, y0, y1)
    raise RuntimeError('failed to find bracket')
    
# %% Solvers

@njit_alternative
def false_position(f, x0, x1, y0=None, y1=None, x=None,
                   xtol=1e-6, ytol=5e-8, args=(), maxiter=50,
                   checkroot=True, checkbounds=True):
    """False position solver."""
    if x is None: x = 1e32
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < 0.:  x1, y1, x0, y0 = x0, y0, x1, y1
    abs_ = abs
    err0 = abs_(y0)
    err1 = abs_(y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    if checkbounds and y0 * y1 > 0.:
        raise ValueError('f(x0) and f(x1) must have opposite signs')
    dx = x1 - x0
    df = - y0
    false_position_iter = utils.false_position_iter
    if utils.not_within_bounds(x, x0, x1):
        x = false_position_iter(x0, x1, dx, y0, y1, df, x0)
    for iter in range(maxiter):
        y = f(x, *args)
        if y > ytol:
            x1 = x
            err = y1 = y
        elif y < -ytol:
            x0 = x
            y0 = y
            err = df = -y0
        else: return x
        if err < err_best:
            err_best = err
            x_best = x
        dx = x1 - x0
        if abs_(dx) < xtol: break
        x = false_position_iter(x0, x1, dx, y0, y1, df, x)
    if x_best != x: f(x, *args)
    utils.raise_root_error(checkroot and err_best < ytol)
    return x_best

@njit_alternative
def bisection(f, x0, x1, y0=None, y1=None, x=None, xtol=1e-6, ytol=5e-8, args=(),
              maxiter=50, checkroot=True, checkbounds=True):
    """Bisection solver."""
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < 0.:  x1, y1, x0, y0 = x0, y0, x1, y1
    dx = x1 - x0
    abs_ = abs
    err0 = abs_(y0)
    err1 = abs_(y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    if checkbounds and y0 * y1 > 0.:
        raise ValueError('f(x0) and f(x1) must have opposite signs')
    bisect = utils.bisect
    if x is None: x = bisect(x0, x1)
    nytol = -ytol
    for iter in range(maxiter):
        y = f(x, *args)
        if y > ytol:
            x1 = x
            err = y
        elif y < nytol:
            x0 = x
            err = -y
        else: return x
        if err < err_best:
            err_best = err
            x_best = x
        dx = x1 - x0
        if abs_(dx) < xtol: break
        x = bisect(x0, x1)
    if x_best != x: f(x, *args)
    utils.raise_root_error(checkroot and err_best < ytol)
    return x_best

@njit_alternative
def IQ_interpolation(f, x0, x1, y0=None, y1=None, x=None,
                     xtol=1e-6, ytol=5e-8, args=(), maxiter=50,
                     checkroot=True, checkbounds=True):
    """Inverse quadratic interpolation solver."""
    abs_ = abs
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if x is None: x = 1e32
    if y1 < 0.: x1, y1, x0, y0 = x0, y0, x1, y1
    df0 = -y0
    dx = x1 - x0
    if utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx, y0, y1, df0, x0)
    err0 = abs_(y0)
    err1 = abs_(y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    if checkbounds and y0 * y1 > 0.:
        raise ValueError('f(x0) and f(x1) must have opposite signs')
    nytol = -ytol
    for iter in range(maxiter):
        y = f(x, *args)
        if y > ytol:
            y2 = y1
            x2 = x1
            x1 = x
            err = y1 = y
        elif y < nytol:
            y2 = y0
            x2 = x0
            x0 = x
            y0 = y
            err = df0 = -y
        else: return x
        if err < err_best:
            err_best = err
            x_best = x
        dx = x1 - x0
        if abs_(dx) < xtol: break
        x = utils.IQ_iter(y0, y1, y2, x0, x1, x2, dx, df0, x)
    if x_best != x: f(x, *args)
    utils.raise_root_error(checkroot and err_best < ytol)
    return x_best

    
    