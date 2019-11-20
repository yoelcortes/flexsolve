# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:50:26 2019

@author: yoelr
"""
from .exceptions import SolverError

__all__ = ('secant', 'wegstein_secant', 'aitken_secant')

def secant(f, x0, x1, xtol, ytol=5e-8, args=(), maxiter=50):
    """Secant solver."""
    _abs = abs
    y0 = f(x0, *args)
    if _abs(y0) < ytol: return x0
    dx = x1-x0 
    for iter in range(maxiter): 
        y1 = f(x1, *args)
        x1 = x0 - y1*dx/(y1-y0)
        dx = x1-x0 
        if _abs(dx) < xtol or _abs(y1) < ytol: return x1
        x0 = x1
        y0 = y1
    raise SolverError(f'failed to converge after {maxiter} iterations')
    
def wegstein_secant(f, x0, x1, xtol, ytol=5e-8, args=(), maxiter=50):
    """Secant solver with Wegstein acceleration."""
    _abs = abs
    y0 = f(x0, *args)
    if _abs(y0) < ytol: return x0
    y1 = f(x1, *args)
    if _abs(y1) < ytol: return x0
    g0 = x1 - y1*(x1-x0)/(y1-y0)
    y0 = y1
    dx = g0-x1
    x1 = g0
    for iter in range(maxiter):
        y1 = f(x1, *args)
        g1 = x1 - y1*dx/(y1-y0)
        x0 = x1
        try:
            w = dx/(dx-g1+g0)
            x1 = w*g1 + (1.-w)*x1
        except:
            x1 = g1
        dx = x1-x0
        if _abs(dx) < xtol or _abs(y1) < ytol: return x1
        y0 = y1
        g0 = g1
    raise SolverError(f'failed to converge after {maxiter} iterations')
    
def aitken_secant(f, x0, x1, xtol, ytol=5e-8, args=(), maxiter=50):
    """Secant solver with Aitken acceleration."""
    _abs = abs
    y0 = f(x0, *args)
    if _abs(y0) < ytol: return x0
    dx = x1-x0
    for iter in range(maxiter):
        y1 = f(x1, *args)
        if y1 == y0: return x1
        x0 = x1 - y1*dx/(y1-y0) # x0 = g
        dx = x0-x1
        if _abs(dx) < xtol or _abs(y1) < ytol: return x0
        y0 = y1
        y1 = f(x0, *args)
        if y1 == y0: return x0
        x2 = x0 - y1*dx/(y1-y0) # x2 = gg
        if _abs(dx) < xtol or _abs(y1) < ytol: return x2
        dx = x1 - x0 # x - g
        denominator = x2 + dx - x0
        x1 = x1 - dx**2./denominator if denominator else x2
        dx = x1 - x0
        y0 = y1
    raise SolverError(f'failed to converge after {maxiter} iterations')