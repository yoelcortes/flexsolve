# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:56:59 2020

@author: yoelr
"""
import numpy as np
from numba.extending import register_jitable
from . import utils


__all__ = ('fixed_point',
           'conditional_fixed_point',
           'wegstein',
           'conditional_wegstein',
           'aitken',
           'conditional_aitken',
) 

iteration_history = []

class CachedFunction:
    __slots__ = ('f', 'inputs', 'outputs')
    
    def __init__(self, f, inputs=None, outputs=None):
        self.f = f
        self.inputs = [] if inputs is None else inputs
        self.outputs = [] if inputs is None else outputs
        
    def __call__(self, x, *args, **kwargs):
        self.inputs.append(x)
        y = self.f(x, *args, **kwargs)
        self.outputs.append(y)
        return y

def GPiteration():
    pass

def GPacceleration(
        f, x, xtol=5e-8, args=(), maxiter=50, memory=10, checkiter=True,
        checkconvergence=True, bounds=None, convergenceiter=0, subset=0, *, rtol=0
    ):
    """Iterative fixed-point solver with GP acceleration."""
    x0 = x1 = x
    errors = np.zeros(convergenceiter)
    f = CachedFunction(f)
    fixedpoint_converged = utils.fixedpoint_converged
    for iter in range(maxiter):
        x1 = GPiteration(f, x0, args, memory, bounds)
        e = np.abs(x1 - x0)
        if fixedpoint_converged(x0, e, xtol, rtol, subset): return x1
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
def fixed_point(f, x, xtol=5e-8, args=(), maxiter=50, checkiter=True,
                checkconvergence=True, convergenceiter=0, subset=0, *, rtol=0):
    """Iterative fixed-point solver."""
    x0 = x1 = x
    errors = np.zeros(convergenceiter)
    fixedpoint_converged = utils.fixedpoint_converged
    for iter in range(maxiter):
        x1 = f(x0, *args)
        e = np.abs(x1 - x0)
        if fixedpoint_converged(x0, e, xtol, rtol, subset): return x1
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
             checkconvergence=True, convergenceiter=0, subset=0, *, 
             lb=-float('inf'), ub=float('inf'), rtol=0, exp=1.0):
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
        if fixedpoint_converged(x1, e, xtol, rtol, subset): return g1
        x0 = x1
        x1 = wegstein_iter(x1, dx, g1, g0, lb, ub, exp)
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
def conditional_wegstein(f, x, lb=-float('inf'), ub=float('inf'), exp=0.5):
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
        x1 = wegstein_iter(x1, dx, g1, g0, lb, ub, exp)
        g0 = g1
    return x1

@register_jitable(cache=True)
def aitken(f, x, xtol=5e-8, args=(), maxiter=50, checkiter=True,
           checkconvergence=True, convergenceiter=0, subset=0, *, rtol=0):
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
        if fixedpoint_converged(x, e, xtol, rtol, subset): return g
        gg = f(g, *args)
        dgg_g = gg - g
        if fixedpoint_converged(g, np.abs(dgg_g), xtol, rtol, subset): return gg
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
