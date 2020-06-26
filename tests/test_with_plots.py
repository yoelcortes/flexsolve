# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:12:04 2020

@author: yoelr
"""
import flexsolve as flx 
from scipy import optimize as opt

# %% Profile solvers

if __name__ == '__main__':
    x0, x1 = [-5, 5]
    f = lambda x: x**3 - 40 + 2*x 
    p = flx.Profiler(f)
    x_brentq = opt.brentq(p, x0, x1, xtol=1e-8)
    p.archive('[Scipy] Brent-Q') # Save/archive results with given name
    x_brenth = opt.brenth(p, x0, x1)
    p.archive('[Scipy] Brent-H')
    x_IQ = flx.IQ_interpolation(p, x0, x1)
    p.archive('IQ-interpolation')
    x_false_position = flx.false_position(p, x0, x1)
    p.archive('False position')
    p.plot(r'$f(x) = 0 = x^3 + 2 \cdot x - 40$ where $-5 < x < 5$')
    
    p = flx.Profiler(f)
    x_guess = -5
    x_aitken_secant = flx.aitken_secant(p, x_guess)
    p.archive('Aitken')
    x_secant = flx.secant(p, x_guess)
    p.archive('Secant')
    x_newton = opt.newton(p, x_guess)
    p.archive('[Scipy] Newton')
    p.plot(r'$f(x) = 0 = x^3 + 2 \cdot x - 40$')
    
    # Note that x = 40/x^2 - 2/x is the same
    # objective function as x**3 - 40 + 2*x = 0
    f = lambda x: 40/x**2 - 2/x
    p = flx.Profiler(f)
    x_guess = 5.
    x_wegstein = flx.wegstein(p, x_guess)
    p.archive('Wegstein')
    x_aitken = flx.aitken(p, x_guess)
    p.archive('Aitken')
    p.plot(r'$f(x) = x = \frac{40}{x^2} - \frac{2}{x}$', markbounds=False)
    # ^ Fixed-point iteration is non-convergent for this equation; so we don't include it here
