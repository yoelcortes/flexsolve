# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:55:16 2020
@author: yoelr
"""
from . import utils
import numpy as np
from numpy import linalg
from collections import deque

__all__ = ('as_least_squares_iter',
           'LeastSquaresIteration',
           'LstSqIter',
)

@utils.njitable
def weighted_average(xs, weights):
    weights /= weights.sum()
    return xs @ weights

def compute_weighted_average_by_least_squares(A, xs):
    b = 1e-16 * np.ones(A.shape[0])
    weights = linalg.lstsq(A, b, None)[0]
    return weighted_average(xs, weights)

class LeastSquaresIteration:
    __slots__ = ('guess_history',
                 'error_history', 
                 'N_activate',
                 '_counter', )
    
    def __init__(self, N_history=5, N_activate=20):
        self.guess_history = deque(maxlen=N_history)
        self.error_history = deque(maxlen=N_history)
        self.N_activate = N_activate
        self._counter = 0

    def __call__(self, x, fx):
        guess_history = self.guess_history
        error_history = self.error_history
        guess_history.append(x)
        error_history.append(fx - x)
        if self.active:
            A = np.array(error_history, dtype=float)
            A = A.transpose()
            xs = np.array(guess_history, dtype=float).transpose()
            return compute_weighted_average_by_least_squares(A, xs)
    
    @property
    def active(self):
        active = self._counter == self.N_activate
        if not active: self._counter += 1
        return active
    
    
LstSqIter = LeastSquaresIteration

def fake_least_squares_iter(x, fx): return fx

def as_least_squares_iter(lstsq):
    if lstsq and not isinstance(lstsq, LstSqIter): lstsq = LstSqIter()
    return lstsq