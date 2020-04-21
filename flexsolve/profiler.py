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
        self.xs = np.array(xs, float)
        self.ys = np.array(ys, float)
    
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
    __slots__ = ('f', 'xs', 'ys', 'archives')
    def __init__(self, f, ):
        self.f = f
        self.archives = []
        self.xs = []
        self.ys = []
        
    def __call__(self, x, *args):
        x = copy(x)
        self.xs.append(x)
        y = self.f(x, *args)
        self.ys.append(y)
        return y
    
    def archive(self, name):
        self.archives.append(Archive(name, self.xs, self.ys))
        self.xs = []
        self.ys = []

    def sizes(self):
        return {name: len(archive) for name, archive in self.archives.items()}
    
    def _plot_points(self, offset, step):
        archives = self.archives
        for archive in archives:
            xs = archive.xs
            ys = archive.ys + offset
            plt.scatter(xs, ys, label=f"{archive.name} ({archive.size} iterations)")
            offset -= step

    def plot(self, title=None, args=(), markbounds=True):
        plt.figure()
        archives = self.archives
        archives.sort(key=lambda x: x.size)
        x_mins = np.array([i.x_min for i in archives])
        x_maxs = np.array([i.x_max for i in archives])
        x_min = x_mins.min()
        x_max = x_maxs.max()
        
        dx = (x_max - x_min) / 50
        xs = np.linspace(x_min - dx, x_max + dx)
        f = lambda x: self.f(x, *args)
        ys = np.array([f(x) for x in xs])
        plt.plot(xs, ys, color='grey')
        y_min = ys.min()
        y_max = ys.max()
        N = len(archives)
        offset = (y_max - y_min) / 3
        step = 2 * offset / N
        offset -= step / 2
        self._plot_points(offset, step)
        plt.fill_between(xs, ys - offset, ys + offset,
                         color='grey', alpha=0.25)
        
        
        x_solution = np.mean([i.xs[-1] for i in archives])
        
        y_solution = f(x_solution)
        y_lb, y_ub = plt.ylim()
        
        x_start = archives[0].xs[0]
        if markbounds and np.all(x_min == x_mins) and np.all(x_max == x_maxs):
            plt.vlines([x_min, x_solution, x_max],
                       [y_lb, y_lb, y_lb],
                       [f(x_min), y_solution, f(x_max)],
                       linestyles = 'dashed',
                       color='grey')
            plt.xticks([x_min, x_solution, x_max], 
                       [f'{x_min:.3g}\nlower\nbound',
                        f'{x_solution:.3g}\nsolution',
                        f'{x_max:.3g}\nupper\nbound'])
        elif np.all(np.array([i.xs[0] for i in archives]) == x_start):
            plt.vlines([x_start, x_solution],
                       [y_lb, y_lb],
                       [f(x_start), y_solution],
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
        plt.tick_params(axis='both', which='both', length=0)
        plt.yticks([])
        plt.legend()
        
    
    def __repr__(self):
        return f"{type(self).__name__}({self.f})"