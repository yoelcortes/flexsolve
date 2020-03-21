# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:02:48 2019

@author: yoelr
"""

__all__ = ('Counter',)

class Counter:
    __slots__ = ('msg', 'N')
    def __init__(self, msg=None, N=0):
        self.msg = msg
        self.N = N
        
    def notify(self):
        print(f"{self.msg or 'counter'}: {self.N}")
        
    def restart(self, msg=None, N=0, notify=True):
        if notify: self.notify()
        self.msg = msg or self.msg
        self.N = N
        
    def count(self):
        self.N += 1
        
    def __repr__(self):
        return f"<Counter: msg={repr(self.msg)}, N={self.N}>"
