# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:18:51 2025

@author: yoelr
"""
from scipy.optimize import fminbound

__all__ = ('line_search',)

def line_search(f, x, descent, t0, t1, ttol=1e-3, maxiter=20):
    return fminbound(
        lambda t: f(x + t * descent),
        x1=t0,
        x2=t1,
        xtol=ttol,
        maxfun=maxiter,
    )
    
    
    
    