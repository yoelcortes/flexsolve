# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:40:01 2019

@author: yoelr
"""

__all__ = ('InfeasibleRegion',
           'ConvergenceError',
           'SolverError',)

class InfeasibleRegion(RuntimeError):
    """Runtime error regarding infeasible processes."""
    def __init__(self, region): 
        self.region = region
        super().__init__(region + ' is infeasible')

class ConvergenceError(RuntimeError):
    """RuntimeError regarding convergence problems."""

class SolverError(ConvergenceError):
    """RuntimeError regarding solver exceeding maximum number of iterations."""
    
    def __init__(self, maxiter, x):
        super().__init__(f'failed to converge after {maxiter} iterations')
        self.x = x
        
