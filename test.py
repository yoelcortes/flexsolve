# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:12:04 2020

@author: yoelr
"""
import numpy as np
from flexsolve import (Profiler,
                       IQ_interpolation, bounded_wegstein, bounded_aitken, 
                       wegstein_secant, aitken_secant, secant,
                       wegstein, aitken)
from scipy.optimize import brentq, brenth, newton

def test_function(x):
    return x**3 - 40 + 2*x 

xlim = [-5, 5]

p = Profiler(test_function)
x_brentq = brentq(p, *xlim)
p.archive('Brent-Q')
x_brenth = brenth(p, *xlim)
p.archive('Brent-H')
ytol = abs(test_function(x_brenth)) + 1e-16
x_IQ = IQ_interpolation(p, *xlim, xtol=1e-16, ytol=ytol)
p.archive('IQ-interpolation')
x_wegstein = bounded_wegstein(p, *xlim, xtol=1e-16, ytol=ytol)
p.archive('Wegstein')
x_aitken = bounded_aitken(p, *xlim, xtol=1e-16, ytol=ytol)
p.archive('Aitken')
p.plot()

p = Profiler(test_function)
x_guess = -5
x_wegstein_secant = wegstein_secant(p, x_guess)
p.archive('Wegstein')
x_aitken_secant = aitken_secant(p, x_guess)
p.archive('Aitken')
x_secant = secant(p, x_guess)
p.archive('Secant')
x_newton = newton(p, x_guess)
p.archive('Newton')
p.plot()

def iter_test_function(x):
    if abs(x) < 1e-3:
        x = 1e-3
    return 40/x**2 - 2/x

p = Profiler(iter_test_function)
x_guess = np.array([5], dtype=float)
x_wegstein = wegstein(p, x_guess)
p.archive('Wegstein')
x_aitken = aitken(p, x_guess)
p.archive('Aitken')
p.plot(markbounds=False)