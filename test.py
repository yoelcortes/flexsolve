# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:13:30 2019

@author: yoelr
"""
from math import log, exp
from scipy.optimize import brentq, brenth
from flexsolve import Profiler, false_position, IQ_interpolation, \
                      bounded_aitken, bounded_wegstein
import matplotlib.pyplot as plt
import numpy as np

f = lambda x: x**2 + x - 0.1*x**3 - 1 + log(x) - exp(-0.1*x)
x0 = 0.1
x1 = 10
y0 = f(x0)
y1 = f(x1)
x = 20
yval = 0
xtol = 1e-8
ytol = 1e-8
xs = np.linspace(x0, x1)
ys = np.array([f(x) for x in xs])
plt.plot(xs, ys)

i = 0
colors = ['k', 'b', 'm', 'y', 'r', 'g']
for solver in (false_position, IQ_interpolation, bounded_wegstein, bounded_aitken):
    name = solver.__name__
    pf = Profiler(f, name)
    xval = solver(pf, x0, x1, y0, y1, x, yval, xtol, ytol)
    c = colors[i]
    plt.plot(xs, ys+i, color=c)
    plt.scatter(pf.xs, np.array(pf.ys) + i, color=c, label=name)
    plt.text(x1+0.1, f(x1)+i, f"{name} [N={pf.counter.N}]", color=c)
    i += 1

for brent in (brentq, brenth):
    name = brent.__name__
    pf = Profiler(f, name)
    xval = brent(pf, x0, x1, xtol=xtol)
    c = colors[i]
    plt.plot(xs, ys+i, color=colors[i])
    plt.scatter(pf.xs, np.array(pf.ys) + i, color=c, label=name)
    plt.text(x1+0.1, f(x1)+i, f"{name} [N={pf.counter.N - 2}]", color=c)
    i += 1
    
plt.axvline(x=xval, color='grey')