# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:38:20 2020

@author: yrc2
"""
import pytest
from .test_fixedpoint_array_solvers import test_fixedpoint_array_solvers
from .test_scalar_solvers import test_scalar_solvers

@pytest.mark.skip(reason="Runs all tests")
def test_flexsolve():
    test_fixedpoint_array_solvers()
    test_scalar_solvers()