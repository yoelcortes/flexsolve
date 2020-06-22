# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:35:01 2019

@author: yoelr
"""
import numpy as np
from .jit_speed import njit_alternative
from . import utils
__all__ = ('false_position', 'bisection', 'bounded_wegstein',
           'bounded_aitken', 'IQ_interpolation', 'find_bracket')

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
    raise Exception('failed to find bracket')
    
# %% Solvers

@njit_alternative
def false_position(f, x0, x1, y0=None, y1=None, x=None, xtol=1e-8, ytol=5e-8, args=()):
    """False position solver."""
    abs_ = abs
    if x is None: x = abs_(x0) + abs_(x1)
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < 0.:  x1, y1, x0, y0 = x0, y0, x1, y1
    dx = x1 - x0
    df = - y0
    err0 = abs_(y0)
    err1 = abs_(y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    false_position_iter = utils.false_position_iter
    if utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx, y0, y1, df, x0)
    while abs_(dx) > xtol:
        y = f(x, *args)
        if y > ytol:
            x1 = x
            err = y1 = y
        elif y < -ytol:
            x0 = x
            y0 = y
            err = df = -y0
        else: return x
        dx = x1 - x0
        x = false_position_iter(x0, x1, dx, y0, y1, df, x)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

@njit_alternative
def bisection(f, x0, x1, y0=None, y1=None, xtol=1e-6, ytol=1e-6, args=()):
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
    bisect = utils.bisect
    x = bisect(x0, x1)
    nytol = -ytol
    while abs_(dx) > xtol:
        y = f(x, *args)
        if y > ytol:
            x1 = x
            err = y
        elif y < nytol:
            x0 = x
            err = -y
        else: return x
        dx = x1 - x0
        x = bisect(x0, x1)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

@njit_alternative
def IQ_interpolation(f, x0, x1, y0=None, y1=None, x=None, xtol=1e-8, ytol=5e-8, args=()):
    """Inverse quadratic interpolation solver."""
    abs_ = abs
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if x is None: x = abs_(x0) + abs_(x1)
    if y1 < 0.:  x1, y1, x0, y0 = x0, y0, x1, y1
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
    nytol = -ytol
    while abs_(dx) > xtol:
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
        dx = x1 - x0
        x = utils.IQ_iter(y0, y1, y2, x0, x1, x2, dx, df0, x)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

@njit_alternative
def bounded_wegstein(f, x0, x1, y0=None, y1=None, x=None, xtol=1e-8, ytol=5e-8, args=()):
    """False position solver with Wegstein acceleration."""
    abs_ = abs
    if x is None: x = abs_(x0) + abs_(x1)
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < 0.:  x1, y1, x0, y0 = x0, y0, x1, y1
    dx = x1 - x0
    df = -y0
    err0 = abs_(y0)
    err1 = abs_(y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    if utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx, y0, y1, df, x0)
    xlast = x
    nytol = -ytol
    y = f(x, *args)
    if y > ytol:
        x1 = x
        err = y1 = y
    elif y < nytol:
        x0 = x
        y0 = y
        err = df = -y
    else:
        return x
    if err < err_best:
        err_best = err
        x_best = x
    dx1x0 = x1-x0
    x = g0 = x0 + df*dx1x0/(y1-y0)
    wegstein_iter = utils.scalar_wegstein_iter
    not_within_bounds = utils.not_within_bounds
    while abs_(dx1x0) > xtol:
        y = f(x, *args)
        if y > ytol:
            x1 = x
            err = y1 = y
        elif y < nytol:
            x0 = x
            y0 = y
            err = df = -y
        else: return x
        dx1x0 = x1-x0
        g1 = x0 + df*dx1x0/(y1-y0)
        dx = x - xlast
        xlast = x
        x = wegstein_iter(x, dx, g1, g0)
        if not_within_bounds(x, x0, x1): x = g0 = g1
        g1 = g0
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    f(x_best, *args)
    return x

@njit_alternative
def bounded_aitken(f, x0, x1, y0=None, y1=None, x=None, xtol=1e-8, ytol=5e-8, args=()):
    """False position solver with Aitken acceleration."""
    abs_ = abs
    if x is None: x = abs_(x0) + abs_(x1)
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    if y1 < 0.:  x1, y1, x0, y0 = x0, y0, x1, y1
    dx1 = x1 - x0
    df = -y0
    err0 = abs_(y0)
    err1 = abs_(y1)
    if err0 < err1:
        x_best = x0
        err_best = err0
    else:
        x_best = x1
        err_best = err1
    if utils.not_within_bounds(x, x0, x1):
        x = utils.false_position_iter(x0, x1, dx1, y0, y1, df, x0)
    aitken_iter = utils.scalar_aitken_iter
    bisect = utils.bisect
    not_within_bounds = utils.not_within_bounds
    nytol = -ytol
    while abs_(dx1) > xtol:
        y = f(x, *args)
        if y > ytol:
            x1 = x
            err = y1 = y
        elif y < nytol:
            x0 = x
            y0 = y
            err = df = -y
        else: 
            return x
        if err < err_best:
            err_best = err
            x_best = x
        dx0 = x1-x0
        g = x0 + df*dx0/(y1-y0)
        if abs_(dx0) < xtol:
            return g
        y = f(g, *args)
        if y > ytol:
            x1 = g
            err = y1 = y
        elif y < nytol:
            x0 = g
            y0 = y
            err = df = -y
        else:
            return g
        dx1 = x1-x0
        gg = x0 + df*dx1/(y1-y0)
        dxg = x - g
        x = aitken_iter(x, gg, dxg, gg - g)
        if not_within_bounds(x, x0, x1): x = bisect(x0, x1)
        if err < err_best:
            err_best = err
            x_best = x
    if x_best != x: f(x_best, *args)
    return x_best

# @njit_alternative
# def solve_bounded_scalar(f, x0, xa, xb, ya=None, yb=None,
#                          xtol=1e-8, ytol=5e-8, args=(), x1=None):
#     """
#     Secant with Aitken acceleration solver that shifts to inverse quadratic
#     interpolation when a guess exceeds the bounds.
#     """
#     abs_ = abs
#     if x1 is None: x1 = x0 + xtol
#     if ya is None: ya = f(xa, *args)
#     if yb is None: yb = f(xb, *args)
#     if yb < 0.:  xb, yb, xa, ya = xa, ya, xb, yb
#     not_within_bounds = utils.not_within_bounds
#     if not_within_bounds(x1, xa, xb):
#         x1 = utils.false_position_iter(xa, xb, xb-xa, ya, yb, -ya, xa)
#     y0 = f(x0, *args)
#     if abs_(y0) < ytol: return x0
#     dx = x1 - x0
#     aitken_iter = utils.scalar_aitken_iter
#     while True:
#         y1 = f(x1, *args)
#         if y1 == y0: break
#         if 0. < y1 < yb:
#             xb = x1; yb = y1
#         elif ya < y1 < 0.:
#             xa = x1; ya = y1
#         else: break
#         x0 = x1 - y1 * dx/(y1 - y0) # x0 = g
#         if not_within_bounds(x0, xa, xb): break
#         dx = x0 - x1
#         if abs_(dx) < xtol or abs_(y1) < ytol: return x0
#         y0 = y1
#         y1 = f(x0, *args)
#         if y1 == y0: break
#         x2 = x0 - y1 * dx / (y1 - y0) # x2 = gg
#         if not_within_bounds(x2, xa, xb): break
#         dx = x1 - x0 # x - g
#         if abs_(dx) < xtol or abs_(y1) < ytol: return x2
#         x1 = aitken_iter(x1, x2, dx, x2 - x0)
#         dx = x1 - x0
#         y0 = y1
#         if 0. < y1 < yb:
#             xb = x1; yb = y1
#         elif ya < y1 < 0.:
#             xa = x1; ya = y1
#         else: break
#     return IQ_interpolation(f, xa, xb, ya, yb, None,
#                             xtol, ytol, args)
    
    
    