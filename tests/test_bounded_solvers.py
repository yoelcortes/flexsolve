# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:12:04 2020

@author: yoelr
"""
import flexsolve as flx 
from numpy.testing import assert_allclose

# %% Profile solvers

# def _generate_assertions():
#     x0, x1 = [-10, 20]
#     f = lambda x: x**3 - 40 + 2*x
#     print(
# f"""
# x0, x1 = [-10, 20]
# f = lambda x: x**3 - 40 + 2*x
# assert_allclose(flx.IQ_interpolation(f, x0, x1), {flx.IQ_interpolation(f, x0, x1)}, rtol=1e-5)
# assert_allclose(flx.bisection(f, x0, x1), {flx.bisection(f, x0, x1)}, rtol=1e-5)
# assert_allclose(flx.false_position(f, x0, x1), {flx.false_position(f, x0, x1)}, rtol=1e-5)
# assert_allclose(flx.find_bracket(f, -5, 5), {flx.find_bracket(f, -5, 5)}, rtol=1e-5)
# assert_allclose(flx.find_bracket(f, -5, 0), {flx.find_bracket(f, -5, 0)}, rtol=1e-5)
# assert_allclose(flx.find_bracket(f, -10, -5), {flx.find_bracket(f, -10, -5)}, rtol=1e-5)
# assert_allclose(flx.find_bracket(f, 5, 10), {flx.find_bracket(f, 5, 10)}, rtol=1e-5)
# assert_allclose(flx.find_bracket(f, 10, 5), {flx.find_bracket(f, 10, 5)}, rtol=1e-5)
# """
# )
    
def test_bounded_solvers():
    x0, x1 = [-10, 20]
    f = lambda x: x**3 - 40 + 2*x
    assert_allclose(flx.IQ_interpolation(f, x0, x1), 3.225240462791775, rtol=1e-5)
    assert_allclose(flx.bisection(f, x0, x1), 3.225240461761132, rtol=1e-5)
    assert_allclose(flx.false_position(f, x0, x1), 3.2252404627266342, rtol=1e-5)
    assert_allclose(flx.find_bracket(f, -5, 5), (-5, 5, -175, 95), rtol=1e-5)
    assert_allclose(flx.find_bracket(f, -5, 0), (-5, 10.0, -175, 980.0), rtol=1e-5)
    assert_allclose(flx.find_bracket(f, -10, -5), (-10, 5.0, -1060, 95.0), rtol=1e-5)
    assert_allclose(flx.find_bracket(f, 5, 10), (-6.073446327683616, 10, -276.1765907762578, 980), rtol=1e-5)
    assert_allclose(flx.find_bracket(f, 10, 5), (-6.073446327683616, 10, -276.1765907762578, 980), rtol=1e-5)
    
    