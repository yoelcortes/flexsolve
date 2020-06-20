# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:43:30 2020

@author: yoelr
"""


def __getattr__(name):
    from flexsolve import speed_up
    import sys 
    
    speed_up()
    dct = sys.modules[__name__].__dict__
    if name in dct:
        return dct[name]
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")