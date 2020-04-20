# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:12:04 2020

@author: yoelr
"""

from math import log
from flexsolve import IQ_interpolation, bounded_wegstein, bounded_aitken, Profiler
from scipy.optimize import brentq, brenth

def test_function(x):
    return x**3 - 40 + 1.3*x + log(max(x, 1)) + 2*(0.1*x)

xlim = [0, 5]

p = Profiler(test_function)
x_IQ = IQ_interpolation(p, *xlim)
p.archive('IQ-interpolation')
x_brentq = brentq(p, *xlim)
p.archive('Brent-Q')
x_brenth = brenth(p, *xlim)
p.archive('Brent-H')
x_wegstein = bounded_wegstein(p, *xlim)
p.archive('Wegstein')
x_aitken = bounded_aitken(p, *xlim)
p.archive('Aitken')

p.plot()