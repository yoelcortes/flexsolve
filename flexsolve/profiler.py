# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:09:53 2019

@author: yoelr
"""
from .counter import Counter

__all__ = ('Profiler',)

class Profiler:
    __slots__ = ('f', 'xs', 'ys', 'counter')
    def __init__(self, f, msg=None):
        self.f = f
        self.counter = Counter(msg)
        self.xs = []
        self.ys = []
        
    def __call__(self, x, *args):
        self.counter.count()
        self.xs.append(x)
        y = self.f(x, *args)
        self.ys.append(y)
        return y
    
    def __repr__(self):
        return f"{type(self).__name__}({self.f})"