# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:50:56 2020

@author: yrc2
"""
from collections.abc import Mapping
from .problem import Problem
import pandas as pd
import numpy as np

__all__ = ('ProblemList',)

class ProblemList(list):
    
    def add_problem(self, f=None, cases=()):
        if not f: return lambda f: self.add_problem(f, cases)
        problem = Problem(f, cases)
        self.append(problem)
        return problem
    
    def profiles_list(self, solver, tol, kwargs):
        if isinstance(kwargs, Mapping):
            return [i.profile_solver(solver, tol, kwargs) for i in self]
        else:
            return [i.profile_solver(solver, tol, j) for i,j in zip(self, kwargs)]
    
    def profiles_dict(self, solver, tol, kwargs):
        if isinstance(kwargs, Mapping):
            return {i.name: i.profile_solver(solver, tol, kwargs) for i in self}
        else:
            return {i.name: i.profile_solver(solver, tol, j) for i,j in zip(self, kwargs)}
        
    def results_df(self, solvers, tol, solver_kwargs=None, solver_names=None):
        solver_problem_profiles = [self.profiles_list(i, tol, j) for i,j in zip(solvers, solver_kwargs)]
        problem_names = [i.name for i in self]
        problem_fields = ['Iterations', 'Passed', 'Failed']
        multi_index = pd.MultiIndex.from_product([problem_names, problem_fields], names=['Problem', 'Summary'])
        columns = solver_names
        get_column = lambda x: sum([(i.size(), len(i.passed_cases), len(i.failed_cases)) for i in x], ())
        data = [get_column(i) for i in solver_problem_profiles]
        data = np.array(data).transpose()
        return pd.DataFrame(data, columns=columns, index=multi_index)
    
    def summary_df(self, solvers, tol, solver_kwargs=None, solver_names=None):
        problem_profiles = [self.profiles_list(i, tol, j) for i,j in zip(solvers, solver_kwargs)]
        passed_cases = [sum([len(i.passed_cases) for i in j]) for j in problem_profiles]
        failed_cases = [sum([len(i.failed_cases) for i in j]) for j in problem_profiles]
        failed_problems = [sum([bool(i.failed_cases) for i in j]) for j in problem_profiles]
        data = np.array([passed_cases, failed_cases, failed_problems])
        return pd.DataFrame(data, columns=solver_names, index=['Passed cases', 'Failed cases', 'Failed problems'])
    
    def __repr__(self):
        return f"{type(self).__name__}({list.__repr__(self)})"