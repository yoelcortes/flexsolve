# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:41:04 2020

@author: yoelr
"""
from .jit_speed import njitable
from collections.abc import Iterable
import numpy as np

np.seterr(divide='raise', invalid='raise')

@njitable
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

def get_wegstein_iter_function(x):
    return array_wegstein_iter if isinstance(x, Iterable) else scalar_wegstein_iter

def get_aitken_iter_function(x):
    return array_aitken_iter if isinstance(x, Iterable) else scalar_aitken_iter

@njitable
def array_wegstein_iter(x, dx, g1, g0):
    denominator = dx-g1+g0
    mask = np.logical_and(np.abs(denominator) > 1e-16, np.abs(dx) < 1e16)
    w = np.ones_like(dx)
    w[mask] = dx[mask]/denominator[mask]
    return w*g1 + (1.-w)*x

@njitable
def scalar_wegstein_iter(x, dx, g1, g0):
    denominator = dx-g1+g0
    if abs(denominator) > 1e-16 and abs(dx) < 1e16:
        w = dx / denominator
        x = w*g1 + (1.-w)*x
    else:
        x = g1
    return x
        
@njitable
def array_aitken_iter(x, gg, dxg, dgg_g):
    denominator = dgg_g + dxg
    mask = np.logical_and(np.abs(denominator) > 1e-16, np.abs(dxg) < 1e16)
    x[mask] -= dxg[mask]**2./denominator[mask]
    nmask = np.logical_not(mask)
    x[nmask] = gg[nmask]
    return x

@njitable
def scalar_aitken_iter(x, gg, dxg, dgg_g):
    denominator = dgg_g + dxg
    if abs(denominator) > 1e-16 and abs(dxg) < 1e16:
        x -= dxg**2. / denominator 
    else:
        x = gg
    return x

@njitable
def not_within_bounds(x, x0, x1):
    return not (x0 < x < x1 or x1 < x < x0)

@njitable
def iteration_is_getting_stuck(x, xlast, dx, r=0.1):
    return abs((x - xlast) / dx) < r

@njitable
def bisect(x0, x1):
    return (x0 + x1) / 2.0

@njitable
def false_position_iter(x0, x1, dx, y0, y1, yval, df, xlast):
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

@njitable
def IQ_iter(y0, y1, y2, yval, x0, x1, x2, dx, df0, xlast):
    df1 = yval - y1
    df2 = yval - y2
    d01 = df0-df1
    d02 = df0-df2
    d12 = df1-df2
    ds = np.array([d01, d02, d12])
    if (np.abs(ds) > 1e-16).all():
        df0_d12 = df0 / d12
        df1_d02 = df1 / d02
        df2_d01 = df2 / d01
        x = x0*df1_d02*df2_d01 - x1*df0_d12*df2_d01 + x2*df0_d12*df1_d02
        if not_within_bounds(x, x0, x1):
            x = bisect(x0, x1)
    else:
        x = false_position_iter(x0, x1, dx, y0, y1, yval, df0, xlast)
    return x