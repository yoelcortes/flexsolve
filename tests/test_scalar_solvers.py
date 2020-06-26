# -*- coding: utf-8 -*-
# Flexsolve: Flexible function solvers.
# Copyright (C) 2019-2020, Yoel Cortes-Pena <yoelcortes@gmail.com>, Caleb Bell <<Caleb.Andrew.Bell@gmail.com>
# 
# This module is under the MIT open-source license. See 
# github.com/yoelcortes/flexsolve/blob/master/LICENSE.txt
# for license details.
"""
"""
from math import log, exp, erf, pi, sin, cos
import numpy as np
import flexsolve as flx

test_problems = flx.ProblemList()
add_problem = test_problems.add_problem

### Known problems ###

@add_problem(cases=[6.25 + 5.0, 6.25 - 1.0, 6.25 + 0.1])
def newton_baffler(x):
    x = x - 6.25
    if x < -0.25:
        return 0.75*x- 0.3125
    elif x < 0.25:
        return 2.0*x
    else:
        return 0.75*x + 0.3125

@add_problem(cases=[2, 0.5, 4])
def flat_stanley(x):
    if x == 1:
        return 0
    else:
        factor = (-1.0 if x < 1.0 else 1.0)
        return factor*exp(log(1000) + log(abs(x - 1.0)) - 1.0/(x - 1.0)**2)

@add_problem(cases=[0.01, -0.25])
def newton_pathological(x):
    if x == 0.0:
        return 0.0
    else:
        factor = (-1.0 if x < 0.0 else 1.0)
        return factor*abs(x)**(1.0/3.0)*exp(-x**2)

@add_problem(cases=[1., -0.14, 0.041])
def repeller(x):
     return 20*x/(100*x**2 + 1)
    
@add_problem(cases=[.25, 5., 1.1])
def pinhead(x):
    return (16 - x**4)/(16*x**4 + 0.00001)
    
@add_problem(cases=[100., 1.])
def lazy_boy(x):
    return 0.00000000001*(x - 100)

@add_problem(cases=[0., 5+180., 5.])
def kepler(x):
    return pi*(x - 5.0)/180.0 - 0.8*sin(pi*x/180)

@add_problem(cases=[3., -0.5, 0, 2.12742])
def camel(x):
    return 1.0/((x - 0.3)**2 + 0.01) + 1.0/((x - 0.9)**2 + 0.04) + 2.0*x - 5.2

@add_problem(cases=[2., 3.])
def Wallis_example(x):
    return x**3 - 2*x - 5

### Unnamed ###

@add_problem(cases=[0.1])
def zero_test_1(x):
    return sin(x) - x/2
        
@add_problem(cases=[1.])
def zero_test_2(x):
    return 2*x - exp(-x)
        
@add_problem(cases=[1.])
def zero_test_3(x):
    return x*exp(-x)

@add_problem(cases=[1.])
def zero_test_4(x):
    return exp(x) - 1.0/(10.0*x)**2

@add_problem(cases=[1.])
def zero_test_5(x):
    return (x+3)*(x-1)**2

@add_problem(cases=[1.])
def zero_test_6(x):
    return exp(x) - 2 - 1/(10*x)**2 + 2/(100*x)**3
       
@add_problem(cases=[1.])
def zero_test_7(x):
    return x**3

@add_problem(cases=[1.])
def zero_test_8(x):
    return cos(x) - x

@add_problem(cases=[.99, 1.013])
def zero_test_9(x):
    return 10.0**14*(1.0*x**7 - 7.0*x**6 + 21.0*x**5 - 35.0*x**4 + 35.0*x**3 - 21.0*x**2 + 7.0*x - 1.0)
        
@add_problem(cases=[0., 1., 0.5])
def zero_test_10(x):
    return cos(100.0*x) - 4.0*erf(30*x - 10)
        
### From Scipy ###

@add_problem(cases=[3,])
def scipy_test_1(x):
    return x**2 - 2*x - 1

@add_problem(cases=[3,])
def scipy_test_2(x):
    return exp(x) - cos(x)

@add_problem(cases=[-1e8, 1e7])
def scipy_GH5555(x):
    return x - 0.1

@add_problem(cases=[0., 1.])
def scipy_GH5557(x):
    return -0.1 if x < 0.5 else x - 0.6 # Fail at 0 is expected - 0 slope
        
@add_problem(cases=[10*(200.0 - 6.828499381469512e-06) / (2.0 + 6.828499381469512e-06)])
def zero_der_nz_dp(x):
    return (x - 100.0)**2 
    
@add_problem(cases=[0., 0.5])
def scipy_GH8904(x):
    return x**3 - x**2
        
@add_problem(cases=[0., 0.5])
def scipy_GH8881(x):
    return x**(1.00/9.0) - 9**(1.0/9)
        
### From gsl ###
        
@add_problem(cases=[3, 4, -4, -3, -1/3., 1])
def gsl_test_1(x):
    return sin(x)

@add_problem(cases=[0, 3, -3, 0])
def gsl_test_2(x):
    return cos(x)
        
@add_problem(cases=[0.1, 2])
def gsl_test_3(x):
    return x**20 - 1

@add_problem(cases=[-1.0/3, 1])
def gsl_test_4(x):
    return np.sign(x)*abs(x)**0.5

@add_problem(cases=[0, 1])
def gsl_test_5(x):
    return x**2 - 1e-8
        
@add_problem(cases=[0.9995, 1.0002])
def gsl_test_6(x):
    return (x-1.0)**7
        
### From Roots.jl ###

@add_problem(cases=[0, 1])
def roots_test_1(x):
    return abs(x - 0.0)

@add_problem(cases=[0, -1, 1, 21])
def roots_test_2(x):
    return 1024*x**11 - 2816*x**9 + 2816*x**7 - 1232*x**5 + 220*x**3 - 11*x

@add_problem(cases=[0, -1, 1, 7])
def roots_test_3(x):
    return 512*x**9 - 1024*x**7 + 672*x**5 - 160*x**3 +10*x


def test_scalar_solvers():
    solvers = [flx.secant, flx.aitken_secant, flx.wegstein, flx.aitken]
    solver_names = [i.__name__ for i in solvers]
    kwargs = [{'ytol': 1e-10}, {'ytol': 1e-10}, {'xtol': 1e-10}, {'xtol': 1e-10}]
    summary_values = np.array([[65, 63, 62, 53],
                               [12, 14, 15, 24],
                               [ 6,  8,  8, 12]])
    df_summary = test_problems.summary_df(solvers,
                                          tol=1e-10,
                                          solver_kwargs=kwargs,
                                          solver_names=solver_names)
    assert np.allclose(df_summary.values, summary_values)
   
    
if __name__ == '__main__':
    solvers = [flx.secant, flx.aitken_secant, flx.wegstein, flx.aitken]
    solver_names = [i.__name__ for i in solvers]
    kwargs = [{'ytol': 1e-10}, {'ytol': 1e-10}, {'xtol': 1e-10}, {'xtol': 1e-10}]
    df_results = test_problems.results_df(solvers,
                                      tol=1e-10,
                                      solver_kwargs=kwargs,
                                      solver_names=solver_names)
    df_summary = test_problems.summary_df(solvers,
                                          tol=1e-10,
                                          solver_kwargs=kwargs,
                                          solver_names=solver_names)