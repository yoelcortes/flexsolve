# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:43:20 2020

@author: yoelr
"""
import numpy as np

__all__ = ('safe_divide',)

def safe_divide(a, b):
    with np.errstate(divide='raise'): return a / b
    
    