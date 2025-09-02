# -*- coding: utf-8 -*-
"""
Line search utilities for simultaneous correction methods.

"""
from scipy.optimize import fminbound

__all__ = ('exact_line_search', 'inexact_line_search')

def exact_line_search(f, x, correction, t0=1e-9, t1=2, ttol=1e-6, maxiter=20,
                      full_output=True):
    return fminbound(
        lambda t: f(x + t * correction),
        x1=t0,
        x2=t1,
        xtol=ttol,
        maxfun=maxiter,
        full_output=full_output,
    )
    
def inexact_line_search(
        f, x, correction, t0=1e-6, t1=2, ttol=1e-3, maxiter=100,
        fx=None, tguess=None, rho=0.618,
    ):
    # Find t such that f(x + t * correction) - fx < 0
    if tguess is None or not t0 < tguess < t1:
        tguess = 0.5 * (t1 + t0)
        fx = None
    if fx is None: 
        fx = f(x)
    x1 = x + t1 * correction
    ft1 = f(x1)
    if ft1 > fx:
        xguess = x + tguess * correction
        ftguess = f(xguess)
        if ftguess > fx:
            for i in range(maxiter):
                t = rho * tguess + (1 - rho) * t0
                xt = x + t * correction
                ft = f(xt)
                if ft < fx: return t, xt, ft
                if abs(t - tguess) < ttol: break
                tguess = t
            x0 = x + t0 * correction
            ft0 = f(x0)
            if ft0 >= fx:
                raise ValueError(
                    'line search could not find improvement over reference point'
                )
            return t0, x0, ft0
        else:
            for i in range(maxiter):
                t = (1 - rho) * tguess + rho * t1
                xt = x + t * correction
                ft = f(xt)
                if ft < ftguess: return t, xt, ft
                if abs(t - tguess) < ttol: break
                t1 = t
            xguess = x + tguess * correction
            ftguess = f(xguess)
            return tguess, xguess, ftguess
    else:
        xguess = x + tguess * correction
        ftguess = f(xguess)
        if ftguess < ft1:
            tnext = tguess
            for i in range(maxiter):
                t = (1 - rho) * tnext + rho * t1
                xt = x + t * correction
                ft = f(xt)
                if ft < ftguess: return t, xt, ft
                if abs(t - tguess) < ttol: break
                tnext = t
            return tguess, xguess, ftguess
        else:
            return t1, x1, ft1
        
        
        
            