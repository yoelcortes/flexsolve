# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:25:27 2020

@author: yoelr
"""
from numba import njit
from . import fast
import sys

__all__ = ('njitable', 'njit_alternative', 'speed_up')

#: All njitable functions
njitables = []

#: All njitable functions that are not replaced in the origin module
njit_alternatives = []

def njitable(f=None, **options):
    """
    Decorate function as njitable. All 'njitable' functions must be 
    compilable by Numba's njit decorator.
    
    Notes
    -----
    When `flexsolve.speed_up` is run, all njitable functions are compiled.
    
    """
    if not f: return lambda f: njitable(f, replace, **options)
    njitables.append((f, options))
    return f

def njit_alternative(f=None, **options):
    """
    Decorate function as njit-alternative. All 'njit_alternative' functions
    must be compilable by Numba's njit decorator. All njit-alternative 
    functions are saved in the `flexsolve.fast` module.
    
    Notes
    -----
    When `flexsolve.speed_up` is run, all njit-alternative functions in 
    the `flexsolve.fast` module are compiled.
    
    """
    if not f: return lambda f: njitable(f, **options)
    njit_alternatives.append((f, options))
    setattr(fast, f.__name__, f)
    return f

def speed_up():
    """
    Speed up simulations by jit compiling all functions registered as
    'njitable'.
    
    See also
    --------
    njit_alternative
    njitable
    
    """
    setfield = setattr
    for f, options in njitables:
        setfield(sys.modules[f.__module__], f.__name__, njit(f, **options))
    njitables.clear()
    for f, options in njit_alternatives:
        setfield(fast, f.__name__,  njit(f, **options))
    njit_alternatives.clear()