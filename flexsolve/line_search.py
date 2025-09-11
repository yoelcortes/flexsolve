# -*- coding: utf-8 -*-
"""
Line search utilities for simultaneous correction methods.

"""
import numpy as np
from typing import NamedTuple
from scipy.optimize import fminbound

__all__ = (
    'exact_line_search',
    'inexact_line_search', 
    'LineSearchResult'
)


class LineSearchResult(NamedTuple):
    t: float #: step length
    x: np.ndarray # data point
    r: float # residual


def exact_line_search(f, x, correction, t0=1e-9, t1=2, ttol=1e-6, maxiter=20):
    t, f, *_ = fminbound(
        lambda t: f(x + t * correction),
        x1=t0,
        x2=t1,
        xtol=ttol,
        maxfun=maxiter,
        full_output=True,
    )
    return LineSearchResult(t, x + t * correction, f)
    
def inexact_line_search(
        f, x, correction, t0=1e-6, t1=1.1, ttol=1e-3, maxiter=10,
        fx=None, tguess=None, rho=0.618,
    ):
    # Find t such that f(x + t * correction) - fx < 0
    if tguess is None or not t0 < tguess < t1: tguess = 0.5 * (t1 + t0)
    if fx is None: fx = f(x)
    x0 = x + t0 * correction
    ft0 = f(x0)
    x1 = x + t1 * correction
    ft1 = f(x1)
    if ft1 > fx:
        xguess = x + tguess * correction
        ftguess = f(xguess)
        if ftguess < fx: # Done!
            best = LineSearchResult(tguess, xguess, ftguess)
        elif ft1 > ft0: # Move towards x0.
            best = LineSearchResult(t0, x0, ft0)
            for i in range(maxiter):
                t = rho * tguess + (1 - rho) * t0
                xt = x + t * correction
                ft = f(xt)
                if ft < best.f: best = LineSearchResult(t, xt, ft)
                if abs(t - tguess) < ttol: break
                tguess = t
        elif ftguess < ft1: # No where to go; moving forward is risky
            best = LineSearchResult(tguess, xguess, ftguess)
        else: # Hope for the best
            best = LineSearchResult(t1, x1, ft1)
    else:
        xguess = x + tguess * correction
        ftguess = f(xguess)
        if ftguess < ft1: # Stick with guess
            best = LineSearchResult(tguess, xguess, ftguess)
        else: # Accelerate
            best = LineSearchResult(t1, x1, ft1)
    return best
        
        
        
            