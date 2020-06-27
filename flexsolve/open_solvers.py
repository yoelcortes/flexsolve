# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:50:26 2019

@author: yoelr
"""
from numba.extending import register_jitable
from flexsolve.bounded_solvers import IQ_interpolation
from flexsolve import utils

__all__ = ('secant', 'aitken_secant')

@register_jitable(cache=True)
def secant(f, x0, x1=None, xtol=0., ytol=5e-8, args=(), maxiter=50,
           checkroot=False, checkiter=True):
    """Secant solver."""
    if checkroot: utils.check_tols(xtol, ytol)
    if x1 is None: x1 = x0 + 1e-5
    y0 = f(x0, *args)
    abs_ = abs
    if abs_(y0) < ytol: return x0
    dx = x1-x0 
    xi0 = x0
    for iter in range(maxiter): 
        y1 = f(x1, *args)
        xtol_satisfied = abs_(dx) < xtol
        ytol_satisfied = abs_(y1) < ytol
        if ytol_satisfied or xtol_satisfied: 
            if checkroot and ytol_satisfied and xtol_satisfied: return x1
            else: return x1
        if y1*y0 < 0.: 
            return IQ_interpolation(f, xi0, x1, y0, y1, None,
                                    xtol, ytol, args, maxiter-iter,
                                    checkroot, checkiter, False)
        if y1 == y0:
            if checkroot or checkiter: utils.raise_tol_error()
            else: return x1
        xi0 = x1
        x1 = x0 - y1*dx/(y1-y0)
        dx = x1-x0 
        x0 = x1
        y0 = y1
    if checkiter: utils.raise_iter_error()
    return x1

@register_jitable(cache=True)
def aitken_secant(f, x0, x1=None, xtol=0., ytol=5e-8, args=(), maxiter=50,
                  checkroot=False, checkiter=True):
    """Secant solver with Aitken acceleration."""
    if checkroot: utils.check_tols(xtol, ytol)
    if x1 is None: x1 = x0 + 1e-5
    abs_ = abs
    y0 = f(x0, *args)
    if abs_(y0) < ytol: return x0
    dx = x1-x0
    aitken_iter = utils.scalar_aitken_iter
    oscillating = False
    for iter in range(maxiter):
        y1 = f(x1, *args)
        xtol_satisfied = abs_(dx) < xtol
        ytol_satisfied = abs_(y1) < ytol
        if ytol_satisfied or xtol_satisfied: 
            if checkroot and ytol_satisfied and xtol_satisfied: return x1
            else: return x1
        inbounds = y1*y0 < 0.
        if y1 == y0:
            if inbounds:
                return IQ_interpolation(f, x0, x1, y0, y1, None,
                                        xtol, ytol, args, maxiter-iter,
                                        checkroot, False)
            if checkroot or checkiter: utils.raise_tol_error()
            else: return x1
        if inbounds:
            if oscillating:
                return IQ_interpolation(f, x0, x1, y0, y1, None,
                                        xtol, ytol, args, maxiter-iter,
                                        checkroot, False)
            else:
                oscillating = True
        x0 = x1 - y1*dx/(y1-y0) # x0 = g
        dx = x0-x1
        y0 = y1
        y1 = f(x0, *args)
        xtol_satisfied = abs_(dx) < xtol
        ytol_satisfied = abs_(y1) < ytol
        if ytol_satisfied or xtol_satisfied: 
            if checkroot and ytol_satisfied and xtol_satisfied: return x0
            else: return x0
        inbounds = y1*y0 < 0.
        if y1 == y0:
            if inbounds:
                return IQ_interpolation(f, x1, x0, y0, y1, None,
                                        xtol, ytol, args, maxiter-iter,
                                        checkroot, False)
            if checkroot or checkiter: utils.raise_tol_error()
            else: return x1
        if inbounds:
            if oscillating:
                return IQ_interpolation(f, x1, x0, y0, y1, None,
                                        xtol, ytol, args, maxiter-iter,
                                        checkroot, False)
            else:
                oscillating = True
        x2 = x0 - y1*dx/(y1-y0) # x2 = gg
        dx = x1 - x0 # x - g
        x1 = aitken_iter(x1, x2, dx, x2 - x0)
        dx = x1 - x0
        y0 = y1
    if checkiter: utils.raise_iter_error()
    return x1
