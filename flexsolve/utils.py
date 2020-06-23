# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:41:04 2020

@author: yoelr
"""
from flexsolve.jit_speed import njitable
from numba.extending import overload, register_jitable
from collections.abc import Iterable
from numba import types
import numpy as np

__all__ = ('pick_best_solution', 'wegstein_iter',
           'aitken_iter', 'array_wegstein_iter', 'scalar_wegstein_iter',
           'array_aitken_iter', 'scalar_aitken_iter', 'not_within_bounds',
           'iteration_is_getting_stuck', 'bisect', 'false_position_iter',
           'IQ_iter', 'fixedpoint_converged')

np.seterr(divide='raise', invalid='raise')

@njitable(cache=True)
def pick_best_solution(xys):
    y_best = 1e16
    x_best = 0.
    abs_ = abs
    for x, y in xys:
        y = abs_(y)
        if y < y_best:
            y_best = y
            x_best = x
    return x_best

def wegstein_iter(x, dx, g1, g0):
    if isinstance(x, Iterable):
        return array_wegstein_iter(x, dx, g1, g0)
    else:
        return scalar_wegstein_iter(x, dx, g1, g0)

def aitken_iter(x, gg, dxg, dgg_g):
    if isinstance(x, Iterable):
        return array_aitken_iter(x, gg, dxg, dgg_g)
    else:
        return scalar_aitken_iter(x, gg, dxg, dgg_g)

@overload(wegstein_iter)
def jit_wegstein_iter(x, dx, g1, g0):
    if isinstance(x, types.Array) and x.ndim:
        return array_wegstein_iter  
    else:
        return scalar_wegstein_iter

@overload(aitken_iter)
def jit_aitken_iter(x, gg, dxg, dgg_g):
    if isinstance(x, types.Array) and x.ndim:
        return array_aitken_iter
    else:
        return scalar_aitken_iter

@register_jitable(cache=True)
def array_wegstein_iter(x, dx, g1, g0):
    denominator = dx-g1+g0
    mask = np.logical_and(np.abs(denominator) > 1e-16, np.abs(dx) < 1e16)
    w = np.ones_like(dx)
    w[mask] = dx[mask]/denominator[mask]
    return w*g1 + (1.-w)*x

@register_jitable(cache=True)
def scalar_wegstein_iter(x, dx, g1, g0):
    denominator = dx-g1+g0
    if abs(denominator) > 1e-16 and abs(dx) < 1e16:
        w = dx / denominator
        x = w*g1 + (1.-w)*x
    else:
        x = g1
    return x
        
@register_jitable(cache=True)
def scalar_fixedpoint_converged(dx, xtol):
    return abs(dx) < xtol

@register_jitable(cache=True)
def array_fixedpoint_converged(dx, xtol):
    return (np.abs(dx) < xtol).all()

def fixedpoint_converged(dx, xtol):
    if isinstance(dx, Iterable) and dx.ndim:
        return array_fixedpoint_converged(dx, xtol)
    else:
        return scalar_fixedpoint_converged(dx, xtol)

@overload(fixedpoint_converged)
def fixedpoint_converged(dx, xtol):
    if isinstance(dx, types.Array) and dx.ndim:
        return array_fixedpoint_converged
    else:
        return scalar_fixedpoint_converged

@register_jitable(cache=True)
def array_aitken_iter(x, gg, dxg, dgg_g):
    denominator = dgg_g + dxg
    mask = np.logical_and(np.abs(denominator) > 1e-16, np.abs(dxg) < 1e16)
    x = x.copy()
    x[mask] -= dxg[mask]**2./denominator[mask]
    nmask = np.logical_not(mask)
    x[nmask] = gg[nmask]
    return x

@register_jitable(cache=True)
def scalar_aitken_iter(x, gg, dxg, dgg_g):
    denominator = dgg_g + dxg
    if abs(denominator) > 1e-16 and abs(dxg) < 1e16:
        x -= dxg**2. / denominator 
    else:
        x = gg
    return x

@njitable(cache=True)
def not_within_bounds(x, x0, x1):
    return not (x0 < x < x1 or x1 < x < x0)

@njitable(cache=True)
def iteration_is_getting_stuck(x, xlast, dx, r=0.1):
    return abs((x - xlast) / dx) < r

@njitable(cache=True)
def bisect(x0, x1):
    return (x0 + x1) / 2.0

@njitable(cache=True)
def false_position_iter(x0, x1, dx, y0, y1, df, xlast):
    dy = y1 - y0
    if dy:
        x = x0 + df*dx/dy
        if not_within_bounds(x, x0, x1):
            x = bisect(x0, x1)
        elif iteration_is_getting_stuck(x, xlast, dx):
            x = (x + x0 + x1) / 3.
    else:
        x = bisect(x0, x1)
    return x

@njitable(cache=True)
def IQ_iter(y0, y1, y2, x0, x1, x2, dx, df0, xlast):
    df1 = -y1
    df2 = -y2
    d01 = df0-df1
    d02 = df0-df2
    d12 = df1-df2
    if abs(d01) > 1e-32 and abs(d02) > 1e-32 and abs(d12) > 1e-32:
        df0_d12 = df0 / d12
        df1_d02 = df1 / d02
        df2_d01 = df2 / d01
        x = x0*df1_d02*df2_d01 - x1*df0_d12*df2_d01 + x2*df0_d12*df1_d02    
        if not_within_bounds(x, x0, x1):
            x = bisect(x0, x1)
    else:
        x = false_position_iter(x0, x1, dx, y0, y1, df0, xlast)
    return x

@njitable(cache=True)
def raise_root_error(not_satisfied):
    if not_satisfied:
        raise RuntimeError('root could not be solved within error tolerance')
        
@njitable(cache=True)
def raise_bounds_error(y0, y1):
    if y0 * y1 > 0.:
        raise RuntimeError('f(x0) and f(x1) must have opposite signs')