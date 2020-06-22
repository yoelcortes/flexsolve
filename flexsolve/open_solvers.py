# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:50:26 2019

@author: yoelr
"""
from flexsolve.jit_speed import njit_alternative
from flexsolve import utils

__all__ = ('secant', 'aitken_secant')

@njit_alternative
def secant(f, x0, x1=None, xtol=1e-6, ytol=5e-8, args=(), maxiter=50,
           checkroot=True):
    """Secant solver."""
    if x1 is None: x1 = x0 + xtol
    y0 = f(x0, *args)
    abs_ = abs
    if abs_(y0) < ytol: return x0
    dx = x1-x0 
    for iter in range(maxiter): 
        y1 = f(x1, *args)
        if abs_(y1) < ytol: return x1
        if y1 == y0 or abs_(dx) < xtol: break 
        x1 = x0 - y1*dx/(y1-y0)
        dx = x1-x0 
        x0 = x1
        y0 = y1
    utils.raise_root_error(checkroot)
    return x1

@njit_alternative
def aitken_secant(f, x0, x1=None, xtol=1e-6, ytol=5e-8, args=(), maxiter=50,
                  checkroot=True):
    """Secant solver with Aitken acceleration."""
    if x1 is None: x1 = x0 + xtol
    abs_ = abs
    y0 = f(x0, *args)
    if abs_(y0) < ytol: return x0
    dx = x1-x0
    aitken_iter = utils.scalar_aitken_iter
    for iter in range(maxiter):
        y1 = f(x1, *args)
        if abs_(y1) < ytol: return x1
        if y1 == y0 or abs_(dx) < xtol: break 
        x0 = x1 - y1*dx/(y1-y0) # x0 = g
        dx = x0-x1
        y0 = y1
        y1 = f(x0, *args)
        if abs_(y1) < ytol: return x0
        if y1 == y0 or abs_(dx) < xtol: x1 = x0; break 
        x2 = x0 - y1*dx/(y1-y0) # x2 = gg
        dx = x1 - x0 # x - g
        x1 = aitken_iter(x1, x2, dx, x2 - x0)
        dx = x1 - x0
        y0 = y1
    utils.raise_root_error(checkroot)
    return x1