# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:35:01 2019

@author: yoelr
"""
from typing import Callable, Iterable, Any, Tuple, Optional
import numpy as np
from numba.extending import register_jitable
from . import utils
__all__ = ('false_position', 'bisection', 
           'IQ_interpolation', 'find_bracket')

# %% Tools

@register_jitable(cache=True)
def find_bracket(
        f: Callable, 
        x0: float,
        x1: float,
        y0: float=None, 
        y1: float=None,
        args: Iterable[Any]=(),
        maxiter: int=50,
        tol: float=5e-8) -> Tuple[float, float, float, float]:
    """
    Return a bracket where the objective function, `f`, is 
    certain to have a root.
    
    """
    if y0 is None: y0 = f(x0, *args)
    if y1 is None: y1 = f(x1, *args)
    isfinite = np.isfinite
    abs_ = abs
    bisect = utils.bisect
    for iter in range(maxiter):
        if y1 < y0: x1, y1, x0, y0 = x0, y0, x1, y1
        dx = x1 - x0 
        if isfinite(y0) and isfinite(y1):
            if y0 * y1 <= 0: return (x0, x1, y0, y1)
            if abs_(dx) < tol: return (x0, x1, y0, y1)
            if y1 != y0:
                x = x0 - 2. * y1*dx/(y1-y0)
                y = f(x, *args)
                if y1 < y:
                    y1 = y
                    x1 = x
                    continue
                elif y < y0:
                    y0 = y
                    x0 = x
                    continue
            if y1 < 0.:
                x1 += 2. * dx
                y1 = f(x1, *args)
            elif y0 > 0.:
                x0 -= 2. * dx
                y0 = f(x0, *args)
        else:                
            x = bisect(x0, x1)
            y = f(x, *args)
            if y > 0.:
                x1 = x
                y1 = y
            else:
                x0 = x
                y0 = y
    raise RuntimeError('failed to find bracket')
    
# %% Solvers

@register_jitable(cache=True)
def false_position(
        f: Callable,
        x0: float, 
        x1: float, 
        y0: Optional[float]=None, 
        y1: Optional[float]=None, 
        x: Optional[float]=None,
        xtol: float=0.,
        ytol: float=5e-8,
        args: Iterable[Any]=(), 
        maxiter: int=50,
        checkroot: bool=False, 
        checkiter: bool=True, 
        checkbounds: bool=True) -> float:
    """False position solver."""
    if checkroot: utils.check_tols(xtol, ytol)
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
    if checkbounds: utils.check_bounds(y0, y1)
    dx = x1 - x0
    df = - y0
    false_position_iter = utils.false_position_iter
    if utils.not_within_bounds(x, x0, x1):
        x = false_position_iter(x0, x1, dx, y0, y1, df, x0)
    for iter in range(maxiter):
        y = f(x, *args)
        if y > 0:
            x1 = x
            err = y1 = y
        elif y < 0.:
            x0 = x
            y0 = y
            err = df = -y0
        else:
            return x
        if err < err_best:
            err_best = err
            x_best = x
        dx = x1 - x0
        xtol_satisfied = abs_(dx) < xtol
        ytol_satisfied = err < ytol
        if ytol_satisfied or xtol_satisfied: 
            if checkroot and ytol_satisfied and xtol_satisfied: return x
            else: checkiter = False; break
        x = false_position_iter(x0, x1, dx, y0, y1, df, x)
    if x_best != x: f(x_best, *args)
    if checkiter: utils.raise_iter_error()
    return x_best

@register_jitable(cache=True)
def bisection(
        f: Callable,
        x0: float, 
        x1: float, 
        y0: Optional[float]=None, 
        y1: Optional[float]=None, 
        x: Optional[float]=None,
        xtol: float=0.,
        ytol: float=5e-8,
        args: Iterable[Any]=(), 
        maxiter: int=50,
        checkroot: bool=False, 
        checkiter: bool=True, 
        checkbounds: bool=True) -> float:
    """Bisection solver."""
    if checkroot: utils.check_tols(xtol, ytol)
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
    if checkbounds: utils.check_bounds(y0, y1)
    bisect = utils.bisect
    if x is None: x = bisect(x0, x1)
    for iter in range(maxiter):
        y = f(x, *args)
        if y > 0.:
            x1 = x
            y1 = err = y
        elif y < 0.:
            x0 = x
            y0 = y
            err = -y
        else:
            return x
        if err < err_best:
            err_best = err
            x_best = x
        dx = x1 - x0
        xtol_satisfied = abs_(dx) < xtol
        ytol_satisfied = err < ytol
        if ytol_satisfied or xtol_satisfied: 
            if checkroot and ytol_satisfied and xtol_satisfied: return x
            else: checkiter = False; break
        x = bisect(x0, x1)
    if x_best != x and y0 * y1 > 0.: f(x_best, *args)
    if checkiter: utils.raise_iter_error()
    return x_best

@register_jitable(cache=True)
def IQ_interpolation(
        f: Callable,
        x0: float, 
        x1: float, 
        y0: Optional[float]=None, 
        y1: Optional[float]=None, 
        x: Optional[float]=None,
        xtol: float=0.,
        ytol: float=5e-8,
        args: Iterable[Any]=(), 
        maxiter: int=50,
        checkroot: bool=False, 
        checkiter: bool=True, 
        checkbounds: bool=True) -> float:
    """Inverse quadratic interpolation solver."""
    if checkroot: utils.check_tols(xtol, ytol)
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
    if checkbounds: utils.check_bounds(y0, y1)
    for iter in range(maxiter):
        y = f(x, *args)
        if y > 0.:
            y2 = y1
            x2 = x1
            x1 = x
            err = y1 = y
        elif y < 0.:
            y2 = y0
            x2 = x0
            x0 = x
            y0 = y
            err = df0 = -y
        else:
            return x
        if err < err_best:
            err_best = err
            x_best = x
        dx = x1 - x0
        xtol_satisfied = abs_(dx) < xtol
        ytol_satisfied = err < ytol
        if ytol_satisfied or xtol_satisfied: 
            if checkroot and ytol_satisfied and xtol_satisfied: return x
            else: checkiter = False; break
        x = utils.IQ_iter(y0, y1, y2, x0, x1, x2, dx, df0, x)
    if x_best != x and y0 * y1 > 0.: f(x_best, *args)
    if checkiter: utils.raise_iter_error()
    return x_best
