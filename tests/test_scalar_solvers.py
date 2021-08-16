# -*- coding: utf-8 -*-
# Flexsolve: Flexible function solvers.
# Copyright (C) 2019-2020, Yoel Cortes-Pena <yoelcortes@gmail.com>, Caleb Bell <Caleb.Andrew.Bell@gmail.com>
# 
# This module is under the MIT open-source license. See 
# github.com/yoelcortes/flexsolve/blob/master/LICENSE.txt
# for license details.
"""
"""
import os
from math import log, exp, erf, pi, sin, cos
from numba import njit
import numpy as np
import flexsolve as flx
import pytest

fixedpoint_solvers = [flx.wegstein, flx.aitken]
open_solvers = [flx.secant, flx.aitken_secant] 
solvers = open_solvers + fixedpoint_solvers
solver_names = [i.__name__ for i in solvers]
kwargs = [{'ytol': 1e-11}, {'ytol': 1e-11}, {'xtol': 1e-11}, {'xtol': 1e-11}]
for i in kwargs: i['maxiter'] = 100
test_problems = flx.ProblemList()
def add_problem(f=None, **kwargs):
    if not f: return lambda f: add_problem(f, **kwargs)
    f = njit(f, cache=True)
    return test_problems.add_problem(f, **kwargs)

### Known problems ###

@add_problem(cases=[6.25 + 5.0, 6.25 - 1.0, 6.25 + 0.1])
def newton_baffler(x, fixedpoint=False):
    y = x - 6.25
    if x < -0.25:
        y = 0.75*y- 0.3125
    elif x < 0.25:
        y = 2.0*y
    else:
        y = 0.75*y + 0.3125
    return y + x if fixedpoint else y

@add_problem(cases=[2, 0.5, 4])
def flat_stanley(x, fixedpoint=False):
    if x == 1:
        y = 0
    else:
        factor = (-1.0 if x < 1.0 else 1.0)
        y = factor*exp(log(1000) + log(abs(x - 1.0)) - 1.0/(x - 1.0)**2)
    return y + x if fixedpoint else y

@add_problem(cases=[0.01, -0.25])
def newton_pathological(x, fixedpoint=False):
    if x == 0.0:
        y = 0.0
    else:
        factor = (-1.0 if x < 0.0 else 1.0)
        y = factor*abs(x)**(1.0/3.0)*exp(-x**2)
    return y + x if fixedpoint else y

@add_problem(cases=[1., -0.14, 0.041])
def repeller(x, fixedpoint=False):
    y = 20*x/(100*x**2 + 1)
    return y + x if fixedpoint else y
    
@add_problem(cases=[.25, 5., 1.1])
def pinhead(x, fixedpoint=False):
    y = (16 - x**4)/(16*x**4 + 0.00001)
    return y + x if fixedpoint else y
    
@add_problem(cases=[100., 1.])
def lazy_boy(x, fixedpoint=False):
    y = 0.00000000001*(x - 100)
    return y + x if fixedpoint else y

@add_problem(cases=[0., 5+180., 5.])
def kepler(x, fixedpoint=False):
    y = pi*(x - 5.0)/180.0 - 0.8*sin(pi*x/180)
    return y + x if fixedpoint else y

@add_problem(cases=[3., -0.5, 0, 2.12742])
def camel(x, fixedpoint=False):
    y = 1.0/((x - 0.3)**2 + 0.01) + 1.0/((x - 0.9)**2 + 0.04) + 2.0*x - 5.2
    return y + x if fixedpoint else y

@add_problem(cases=[2., 3.])
def Wallis_example(x, fixedpoint=False):
    y = x*x*x - 2.*x - 5.
    return y + x if fixedpoint else y

named_problems = test_problems[:]  

### Unnamed ###

@add_problem(cases=[0.1])
def zero_test_1(x, fixedpoint=False):
    y = sin(x) - x/2.
    return y + x if fixedpoint else y
        
@add_problem(cases=[1.])
def zero_test_2(x, fixedpoint=False):
    y = 2.*x - exp(-x)
    return y + x if fixedpoint else y
        
@add_problem(cases=[1.])
def zero_test_3(x, fixedpoint=False):
    y = x*exp(-x)
    return y + x if fixedpoint else y

@add_problem(cases=[1.])
def zero_test_4(x, fixedpoint=False):
    a = (10.0*x)
    y = exp(x) - 1.0/(a*a)
    return y + x if fixedpoint else y

@add_problem(cases=[1.])
def zero_test_5(x, fixedpoint=False):
    y = (x+3.)*(x-1.)**2.
    return y + x if fixedpoint else y

@add_problem(cases=[1.])
def zero_test_6(x, fixedpoint=False):
    y = exp(x) - 2. - 1./(10.*x)**2. + 2./(100.*x)**3.
    return y + x if fixedpoint else y
       
@add_problem(cases=[1.])
def zero_test_7(x, fixedpoint=False):
    y = x*x*x
    return y + x if fixedpoint else y

@add_problem(cases=[1.])
def zero_test_8(x, fixedpoint=False):
    y = cos(x) - x
    return y + x if fixedpoint else y

@add_problem(cases=[.99, 1.013])
def zero_test_9(x, fixedpoint=False):
    y = 10.0**14*(1.0*x**7 - 7.0*x**6 + 21.0*x**5 - 35.0*x**4 + 35.0*x**3 - 21.0*x**2 + 7.0*x - 1.0)
    return y + x if fixedpoint else y
        
@add_problem(cases=[0., 1., 0.5])
def zero_test_10(x, fixedpoint=False):
    y = cos(100.0*x) - 4.0*erf(30*x - 10)
    return y + x if fixedpoint else y

other_problems = test_problems[-10:]        

### From Scipy ###

@add_problem(cases=[3,])
def scipy_test_1(x, fixedpoint=False):
    y = x**2 - 2*x - 1
    return y + x if fixedpoint else y

@add_problem(cases=[3,])
def scipy_test_2(x, fixedpoint=False):
    y = exp(x) - cos(x)
    return y + x if fixedpoint else y

@add_problem(cases=[-1e8, 1e7])
def scipy_GH5555(x, fixedpoint=False):
    y = x - 0.1
    return y + x if fixedpoint else y

@add_problem(cases=[0., 1.])
def scipy_GH5557(x, fixedpoint=False):
    y = -0.1 if x < 0.5 else x - 0.6 # Fail at 0 is expected - 0 slope
    return y + x if fixedpoint else y
        
@add_problem(cases=[10*(200.0 - 6.828499381469512e-06) / (2.0 + 6.828499381469512e-06)])
def zero_der_nz_dp(x, fixedpoint=False):
    y = (x - 100.0)**2 
    return y + x if fixedpoint else y
    
@add_problem(cases=[0., 0.5])
def scipy_GH8904(x, fixedpoint=False):
    y = x**3 - x**2
    return y + x if fixedpoint else y
        
@add_problem(cases=[0., 0.5])
def scipy_GH8881(x, fixedpoint=False):
    y = x**(1.00/9.0) - 9**(1.0/9.)
    return y + x if fixedpoint else y
        
scipy_problems = test_problems[-7:]

### From gsl ###
        
@add_problem(cases=[3, 4, -4, -3, -1/3., 1])
def gsl_test_1(x, fixedpoint=False):
    y = sin(x)
    return y + x if fixedpoint else y

@add_problem(cases=[0, 3, -3, 0])
def gsl_test_2(x, fixedpoint=False):
    y = cos(x)
    return y + x if fixedpoint else y
        
@add_problem(cases=[0.1, 2.])
def gsl_test_3(x, fixedpoint=False):
    y = x**20. - 1.
    return y + x if fixedpoint else y

@add_problem(cases=[-1.0/3, 1])
def gsl_test_4(x, fixedpoint=False):
    y = x/abs(x)*abs(x)**0.5
    return y + x if fixedpoint else y

@add_problem(cases=[0, 1])
def gsl_test_5(x, fixedpoint=False):
    y = x*x - 1e-8
    return y + x if fixedpoint else y
        
@add_problem(cases=[0.9995, 1.0002])
def gsl_test_6(x, fixedpoint=False):
    y = (x-1.0)**7.
    return y + x if fixedpoint else y

gsl_problems = test_problems[-6:]

### From Roots.jl ###

@add_problem(cases=[0, 1])
def roots_test_1(x, fixedpoint=False):
    y = abs(x - 0.0)
    return y + x if fixedpoint else y

@add_problem(cases=[0, -1, 1., 21.])
def roots_test_2(x, fixedpoint=False):
    y = 1024.*x**11. - 2816.*x**9. + 2816.*x**7. - 1232.*x**5. + 220.*x**3. - 11.*x
    return y + x if fixedpoint else y

@add_problem(cases=[0, -1, 1, 7])
def roots_test_3(x, fixedpoint=False):
    y = 512.*x**9. - 1024.*x**7. + 672.*x**5. - 160.*x**3. +10.*x
    return y + x if fixedpoint else y

julia_problems = test_problems[-3:]

def test_scalar_solvers():
    if os.environ.get("NUMBA_DISABLE_JIT") == '1':
        summary_values = np.array(
            [[67, 66, 53, 44],
             [10, 11, 24, 33],
             [ 7,  8, 12, 17]]
        )
    else:
        summary_values = np.array(
            [[68, 66, 51, 42],
             [ 9, 11, 26, 35],
             [ 6,  8, 13, 18]]
        )
    summary_array = test_problems.summary_array(solvers, tol=1e-10, solver_kwargs=kwargs)
    assert np.allclose(summary_array, summary_values)
   
# @pytest.mark.slow
# def test_scalar_solvers_with_numba():
#     # This test takes about 15 sec because we are compiling 
#     # every solver-problem version. There is no way to cache all these
#     # due to weakrefs (unless we use dill instead of pickle for numba).
#     summary_values = np.array(
#        [[10, 10,  6,  6],
#         [ 0,  0,  4,  4],
#         [ 0,  0,  2,  2]]
#     )
#     jitted_open_solvers = [njit(i) for i in open_solvers]
#     jitted_fixedpoint_solvers = [njit(i) for i in fixedpoint_solvers]
#     jitted_solvers = jitted_open_solvers + jitted_fixedpoint_solvers
#     results = np.zeros((3, len(jitted_solvers)))
#     abs_ = abs
#     for i, solver in enumerate(jitted_solvers):
#         failed_cases = 0
#         passed_cases = 0
#         failed_problems = 0
#         isfixedpoint = solver in jitted_fixedpoint_solvers
#         args = (isfixedpoint,)
#         kwargsi = kwargs[i]
#         for problem in julia_problems: # Only check a subset for numba
#             f = problem.f
#             problem_failed = False
#             for case in problem.cases:
#                 if isfixedpoint:
#                     # More or less account f(x) = x instead of f(x) = 0
#                     case = f(case) - case 
#                 try:
#                     x = solver(f, case, args=args, **kwargsi)
#                     assert abs_(f(x)) <= 1e-10, "result not within tolerance"
#                 except Exception:
#                     problem_failed = True
#                     failed_cases += 1
#                 else:
#                     passed_cases += 1
#             failed_problems += problem_failed
#         results[:, i] = [passed_cases, failed_cases, failed_problems]
#     assert np.allclose(results, summary_values) 
   
if __name__ == '__main__':
    df_results = test_problems.results_df(solvers,
                                      tol=1e-10,
                                      solver_kwargs=kwargs,
                                      solver_names=solver_names)
    df_summary = test_problems.summary_df(solvers,
                                          tol=1e-10,
                                          solver_kwargs=kwargs,
                                          solver_names=solver_names)