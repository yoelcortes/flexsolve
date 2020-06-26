# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:25:27 2020

@author: yoelr
"""
from numba import njit
from numba.extending import register_jitable
import sys

__all__ = ('njitable', 'speed_up')

#: All njitable functions.
njitables = []

def njitable(f=None, **options):
    """
    Decorate function as njitable. All 'njitable' functions must be 
    compilable by Numba's njit decorator.
    
    Notes
    -----
    When `flexsolve.speed_up` is run, all njitable functions are compiled.
    
    """
    if not f: return lambda f: njitable(f, **options)
    f_jitable = register_jitable(**options)(f) if options else register_jitable(f)
    njitables.append((f, options))
    return f_jitable

def speed_up():
    """
    Speed up simulations by jit compiling all functions registered as
    'njitable'.
    
    See also
    --------
    njitable
    njit_alternative
    
    """
    setfield = setattr
    for f, options in njitables:
        f_jit =  njit(f, **options)
        setfield(sys.modules[f.__module__], f.__name__, f_jit)
    njitables.clear()