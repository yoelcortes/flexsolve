# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:47:51 2019

@author: yoelr
"""
from . import open_solvers
from . import bounded_solvers
from . import fixed_point_solvers
from . import line_search
from . import numerical_analysis
from . import problem_list
from . import profiler
from . import problem
from . import utils

__all__ = (
    *open_solvers.__all__,
    *bounded_solvers.__all__,
    *fixed_point_solvers.__all__,
    *line_search.__all__,
    *numerical_analysis.__all__,
    *problem_list.__all__,
    *problem.__all__,
    *profiler.__all__,
    'utils',
)

from .open_solvers import *
from .bounded_solvers import *
from .fixed_point_solvers import *
from .line_search import *
from .numerical_analysis import *
from .problem_list import *
from .problem import *
from .profiler import *

__version__ = '0.5.8'
