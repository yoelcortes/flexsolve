# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:56:59 2020

@author: yoelr
"""
import numpy as np
from numba.extending import register_jitable
from flexsolve.least_squares_iteration import as_least_squares_iter
from . import utils


__all__ = ('fixed_point',
           'conditional_fixed_point',
           'wegstein',
           'conditional_wegstein',
           'aitken',
           'conditional_aitken',
           'fixed_point_lstsq',
) 

def fixed_point_lstsq(f, x, xtol=5e-8, args=(), maxiter=50, lstsq=True,
                      checkiter=True, checkconvergence=True, convergenceiter=0):
    """The least-squares solution of a matrix of prior iterations is partially
    used to iteratively esmitate the root."""
    lstsq = as_least_squares_iter(lstsq)
    x0 = x1 = x
    errors = np.zeros(convergenceiter)
    fixedpoint_converged = utils.fixedpoint_converged
    for iter in range(maxiter):
        if x0 is None:
            x0 = x1
            x1 = f(x0, *args)
        else:
            try: x1 = f(x0, *args)
            except: # pragma: no cover
                x0 = x1
                x1 = f(x0, *args)
        e = np.abs(x1 - x0)
        if fixedpoint_converged(e, xtol): return x1
        x0 = lstsq(x0, x1)
        if convergenceiter:
            mean = utils.mean(e)
            if iter > convergenceiter and mean > errors.mean():
                if checkconvergence: utils.raise_convergence_error()
                else: return x1
            errors = np.roll(errors, shift=1)
            errors[-1] = mean
    if checkiter: utils.raise_iter_error()
    return x1

@register_jitable(cache=True)
def fixed_point(f, x, xtol=5e-8, args=(), maxiter=50, checkiter=True,
                checkconvergence=True, convergenceiter=0):
    """Iterative fixed-point solver."""
    x0 = x1 = x
    errors = np.zeros(convergenceiter)
    fixedpoint_converged = utils.fixedpoint_converged
    for iter in range(maxiter):
        x1 = f(x0, *args)
        e = np.abs(x1 - x0)
        if fixedpoint_converged(e, xtol): return x1
        if convergenceiter:
            mean = utils.mean(e)
            if iter > convergenceiter and mean > errors.mean():
                if checkconvergence: utils.raise_convergence_error()
                else: return x1
            errors = np.roll(errors, shift=1)
            errors[-1] = mean
        x0 = x1
    if checkiter: utils.raise_iter_error()
    return x1

@register_jitable(cache=True)
def conditional_fixed_point(f, x):
    """Conditional iterative fixed-point solver."""
    x0 = x1 = x
    condition = True
    while condition:
        x1, condition = f(x0)
        x0 = x1
    return x1

@register_jitable(cache=True)
def wegstein(f, x, xtol=5e-8, args=(), maxiter=50, checkiter=True,
             checkconvergence=True, convergenceiter=0):
    """Iterative Wegstein solver."""
    errors = np.zeros(convergenceiter)
    x0 = x
    x1 = g0 = f(x0, *args)
    wegstein_iter = utils.wegstein_iter
    fixedpoint_converged = utils.fixedpoint_converged
    for iter in range(maxiter):
        dx = x1 - x0
        try: g1 = f(x1, *args)
        except: # pragma: no cover
            x1 = g0
            g1 = f(x1, *args)
        e = np.abs(g1 - x1)
        if fixedpoint_converged(e, xtol): return g1
        x0 = x1
        x1 = wegstein_iter(x1, dx, g1, g0)
        g0 = g1
        if convergenceiter:
            mean = utils.mean(e)
            if iter > convergenceiter and mean > errors.mean():
                if checkconvergence: utils.raise_convergence_error()
                else: return x1
            errors = np.roll(errors, shift=1)
            errors[-1] = mean
    if checkiter: utils.raise_iter_error()
    return x1

@register_jitable(cache=True)
def conditional_wegstein(f, x):
    """Conditional iterative Wegstein solver."""
    x0 = x
    g0, condition = f(x0)
    g1 = x1 = g0
    wegstein_iter = utils.wegstein_iter
    while condition:
        try: g1, condition = f(x1)
        except: # pragma: no cover
            x1 = g1
            g1, condition = f(x1)
        g1 = g1
        dx = x1-x0
        x0 = x1
        x1 = wegstein_iter(x1, dx, g1, g0)
        g0 = g1
    return x1

@register_jitable(cache=True)
def aitken(f, x, xtol=5e-8, args=(), maxiter=50, checkiter=True,
           checkconvergence=True, convergenceiter=0):
    """Iterative Aitken solver."""
    gg = x
    errors = np.zeros(convergenceiter)
    aitken_iter = utils.aitken_iter
    fixedpoint_converged = utils.fixedpoint_converged
    for iter in range(maxiter):
        try: g = f(x, *args)
        except: # pragma: no cover
            x = gg
            g = f(x, *args)
        dxg = x - g
        e = np.abs(dxg)
        if fixedpoint_converged(e, xtol): return g
        gg = f(g, *args)
        dgg_g = gg - g
        if fixedpoint_converged(np.abs(dgg_g), xtol): return gg
        x = aitken_iter(x, gg, dxg, dgg_g)
        if convergenceiter:
            mean = utils.mean(e)
            if iter > convergenceiter and mean > errors.mean():
                if checkconvergence: utils.raise_convergence_error()
                else: return x
            errors = np.roll(errors, shift=1)
            errors[-1] = mean
    if checkiter: utils.raise_iter_error()
    return x

@register_jitable(cache=True)
def conditional_aitken(f, x):
    """Conditional iterative Aitken solver."""
    condition = True
    gg = x
    aitken_iter = utils.aitken_iter
    while condition:
        try:
            g, condition = f(x)
        except: # pragma: no cover
            x = gg.copy()
            g, condition = f(x)
        if not condition: return g
        gg, condition = f(g)
        x = aitken_iter(x, gg, x - g, gg - g)
    return x