# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:09:53 2019

@author: yoelr
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

__all__ = ('Profiler',)
            
            
class Archive:
    __slots__ = ('name', 'xs', 'ys')
    
    def __init__(self, name, xs, ys):
        self.name = name
        self.xs = np.array(xs)
        self.ys = np.array(ys)
    
    def __len__(self):
        return len(self.xs)
    
    @property
    def size(self):
        return self.xs.size
    
    @property
    def x_min(self):
        return self.xs.min()
    
    @property
    def x_max(self):
        return self.xs.max()
    
    @property
    def y_min(self):
        return self.ys.min()
    
    @property
    def y_max(self):
        return self.ys.max()
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.name)}, xs={self.xs}, ys={self.ys})"

    def _ipython_display_(self):
        print(f"{type(self).__name__}(\n"
              f"    {repr(self.name)},\n"
              f"    xs={self.xs},\n"
              f"    ys={self.ys}\n"
              ")")


class Profiler:
    __slots__ = ('f', 'xs', 'ys', 'active_archives', 'passed_cases', 'failed_cases')
    def __init__(self, f):
        self.f = f
        self.active_archives = self.passed_cases = []
        self.failed_cases = []
        self.xs = []
        self.ys = []
        
    def __call__(self, x, *args):
        x = copy(x)
        self.xs.append(x)
        y = self.f(x, *args)
        self.ys.append(y)
        return y
    
    def archive(self, name):
        self.active_archives.append(Archive(name, self.xs, self.ys))
        self.xs = []
        self.ys = []

    def archive_case(self, case, failed=False):
        archives = self.failed_cases if failed else self.passed_cases
        archives.append(Archive(case, self.xs, self.ys))
        self.xs = []
        self.ys = []

    def size(self):
        return sum([len(archive) for archive in self.active_archives])

    def sizes(self):
        return {archive.name: len(archive) for archive in self.active_archives}
    
    def activate_failed_archives(self):
        self.active_archives = self.failed_cases
    
    def activate_passed_archives(self):
        self.active_archives = self.passed_cases
    
    def _plot_points(self, rxs, rys, offset, step):
        cycle = plt.rcParams['axes.prop_cycle']
        colors = cycle.by_key()['color']
        archives = self.active_archives
        for color, archive in zip(colors, archives):
            xs = archive.xs
            ys = archive.ys + offset
            plt.scatter(xs, ys, color=color,
                        label=f"{archive.name} ({archive.size} iterations)")
            plt.plot(rxs, rys + offset, color=color, alpha=0.85)
            offset -= step

    def plot(self, title=None, args=(), markbounds=True,
             plot_outside_bounds=True, N=50, shade=True,
             remove_ticks=True):
        plt.figure()
        archives = self.active_archives
        archives.sort(key=lambda x: x.size)
        x_mins = np.array([i.x_min for i in archives])
        x_maxs = np.array([i.x_max for i in archives])
        x_min = x_mins.min()
        x_max = x_maxs.max()
        dx = (x_max - x_min) / 50 if plot_outside_bounds else 0.
        xs = np.linspace(x_min - dx, x_max + dx, N)
        f = lambda x: self.f(x, *args)
        ys = np.array([f(x) for x in xs])
        y_min = ys.min()
        y_max = ys.max()
        N = len(archives)
        X = max(12 - 2*N, 3)
        offset = (y_max - y_min) / X
        step = 2 * offset / N
        offset -= step / 2
        self._plot_points(xs, ys, offset, step)
        if shade:
            plt.fill_between(xs, ys - offset - step, ys + offset + step,
                             color='grey', alpha=0.1)

        x_solution = np.mean([i.xs[-1] for i in archives])        
        y_solution = f(x_solution)
        y_lb, y_ub = plt.ylim()
        
        x_start = archives[0].xs[0]
        if markbounds and np.all(x_min == x_mins) and np.all(x_max == x_maxs):
            plt.vlines([x_min, x_solution, x_max],
                       [y_lb, y_lb, y_lb],
                       [f(x_min) + offset + step, y_solution + offset + step, f(x_max) + offset + step],
                       linestyles = 'dashed',
                       color='grey')
            plt.xticks([x_min, x_solution, x_max], 
                       [f'{x_min:.3g}\nlower\nbound',
                        f'{x_solution:.3g}\nsolution',
                        f'{x_max:.3g}\nupper\nbound'])
        elif np.all(np.array([i.xs[0] for i in archives]) == x_start):
            plt.vlines([x_start, x_solution],
                       [y_lb, y_lb],
                       [f(x_start) + offset + step,
                        y_solution + offset + step],
                       linestyles = 'dashed',
                       color='grey')
            plt.xticks([x_start, x_solution],
                       [f'{x_start:.3g}\nguess',
                        f'{x_solution:.3g}\nsolution'])
        else:
            plt.vlines([x_solution],
                       [y_lb],
                       [y_solution],
                       linestyles = 'dashed',
                       color='grey')
            plt.xticks([x_solution], ['solution'])
        
        if title: plt.title(title)
        plt.ylim([y_lb, y_ub])
        plt.xlim([xs[0], xs[-1]])
        if remove_ticks:
            plt.tick_params(axis='both', which='both', length=0)
            plt.yticks([])
        plt.legend()
        
    
    def __repr__(self):
        return f"{type(self).__name__}({self.f})"
    