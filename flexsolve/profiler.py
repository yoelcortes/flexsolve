# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:09:53 2019

@author: yoelr
"""
import numpy as np
import matplotlib.pyplot as plt

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
        return (f"{type(self).__name__}(\n"
                f"    {repr(self.name)}\n"
                f"    xs={self.xs},\n"
                f"    ys={self.ys}\n"
                 ")")


class Profiler:
    __slots__ = ('f', 'xs', 'ys', 'archives')
    def __init__(self, f):
        self.f = f
        self.archives = []
        self.xs = []
        self.ys = []
        
    def __call__(self, x, *args):
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
            plt.scatter(xs, ys, label=f"{archive.name} ({archive.size})")
            offset -= step

    def plot(self):
        archives = self.archives
        archives.sort(key=lambda x: x.size)
        x_min = min([i.x_min for i in archives])
        x_max = max([i.x_max for i in archives]) 
        y_min = min([i.y_min for i in archives])
        y_max = min([i.y_max for i in archives])
        
        xs = np.linspace(x_min, x_max)
        f = self.f
        ys = [f(x) for x in xs]
        plt.plot(xs, ys, '--', color='grey')
        x_solution = np.mean([i.xs[-1] for i in archives])
        plt.axvline(x=x_solution, color='grey')
        
        N = len(archives)
        offset = (y_max - y_min) / 3
        step = 2 * offset / N
        offset -= step / 2
        self._plot_points(offset, step)
        plt.legend()
        
    
    def __repr__(self):
        return f"{type(self).__name__}({self.f})"