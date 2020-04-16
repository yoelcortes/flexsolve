# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:40:01 2019

@author: yoelr
"""

__all__ = ('SolverError',)

class SolverError(RuntimeError):
    """RuntimeError regarding solvers."""
    
    def __init__(self, maxiter, x):
        super().__init__(f'failed to converge after {maxiter} iterations')
        self.x = x