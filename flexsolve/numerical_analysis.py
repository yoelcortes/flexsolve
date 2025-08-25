# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 16:48:47 2025

@author: yoelr
"""
import numpy as np
from scipy.differentiate import jacobian
from numpy.linalg import cond as matrix_condition_number
from scipy.linalg import block_diag

__all__ = (
    'function_condition_number',
    'independent_functions_condition_number',
    'matrix_condition_number',
)

def norm(x):
    return np.linalg.norm(x, ord=2) if np.ndim(x) else x

def independent_functions_condition_number(*fxs, **jac_kwargs):
    """
    Estimate the condition number of a set of nonlinear functions at a 
    given point using finite-difference Jacobian from SciPy.

    Parameters
    ----------
    fxs : callable
        Pairs of functios and points at which to evaluate the condition number.
        The function is from ℝⁿ to ℝᵐ and must accept a 1D NumPy array.         
    **jac_kwargs : dict, optional
        Additional keyword arguments passed to `scipy.differentiate.jacobian`,
        such as `initial_step`, `order`, `step_direction`, etc.

    Returns
    -------
    kappa : float
        The estimated condition number at point `x`. If `f(x)` has zero norm,
        returns `np.inf`.

    Notes
    -----
    The condition number is computed as:

        κ(x) = ||J(x)|| * ||x|| / ||f(x)||

    where J(x) is the Jacobian of `f` at `x`, and all norms are 2-norms 
    (by default).
    """
    xs = []
    fs = []
    Js = []
    def add(lst, value):
        if np.ndim(value):
            lst.extend(value)
        else:
            lst.append(value)
            
    for f, x in fxs:
        x = np.asarray(x, dtype=float)
        res = jacobian(f, x, **jac_kwargs)
        fx = f(x)
        J = res.df
        add(xs, x)
        add(fs, fx)
        add(Js, J)
        
    J = block_diag(*Js)
    x = np.array(xs)
    fx = np.array(fs)
    norm_x = norm(x)
    norm_fx = norm(fx)
    norm_J = norm(J)
    if norm_fx == 0:
        kappa = np.inf
    else:
        kappa = norm_J * norm_x / norm_fx
    return kappa

def function_condition_number(f, x, **jac_kwargs):
    """
    Estimate the condition number of a nonlinear function at a given point
    using finite-difference Jacobian from SciPy.

    Parameters
    ----------
    f : callable
        A function from ℝⁿ to ℝᵐ. Must accept a 1D NumPy array of shape (n,)
        and return a 1D array of shape (m,).
    x : array_like, shape (n,)
        Point at which to evaluate the condition number.
    **jac_kwargs : dict, optional
        Additional keyword arguments passed to `scipy.differentiate.jacobian`,
        such as `initial_step`, `order`, `step_direction`, etc.

    Returns
    -------
    kappa : float
        The estimated condition number at point `x`. If `f(x)` has zero norm,
        returns `np.inf`.

    Notes
    -----
    The condition number is computed as:

        κ(x) = ||J(x)|| * ||x|| / ||f(x)||

    where J(x) is the Jacobian of `f` at `x`, and all norms are 2-norms 
    (by default).
    """
    x = np.asarray(x, dtype=float)
    # Compute numerical Jacobian using SciPy's differentiate API
    res = jacobian(f, x, **jac_kwargs)
    J = res.df  # The Jacobian matrix (m, n)
    fx = f(x)
    norm_x = norm(x)
    norm_fx = norm(fx)
    norm_J = norm(J)
    if norm_fx == 0:
        kappa = np.inf
    else:
        kappa = norm_J * norm_x / norm_fx
    return kappa


if __name__ == '__main__':
    
    def f0(x0):
        return x0[0] ** 2 - x0[1] ** 0.5 + 5  
    
    def f1(x1):
        return x1[0] ** 0.8 - x1[1] ** 3
    
    def f(x):
        return np.array([f0(x[:2]), f1(x[2:])])
    
    x = np.array([2, 3, 0.5, -1])
    kappa = function_condition_number(f, x)
    kappa_mix = independent_functions_condition_number((f0, x[:2]), (f1, x[2:]))