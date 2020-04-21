========================================================
flexsolve: Flexible function solvers
========================================================
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/yoelcortes/flexsolve/blob/master/LICENSE.txt
   :alt: license


.. contents::

What is flexsolve?
------------------

flexsolve presents a flexible set of function solvers by defining alternative
tolerance conditions for accepting a solution. These solvers also implement
methods like Wegstein and Aitken-Steffensen acceleration to reach solutions
quicker.

Installation
------------

Get the latest version of flexsolve from `PyPI <https://pypi.python.org/pypi/flexsolve/>`__. If you have an installation of Python with pip, simple install it with:

    $ pip install flexsolve

To get the git version, run:

    $ git clone git://github.com/yoelcortes/flexsolve

Documentation
-------------

No extensive documentation is available. However, the parameters of each
solver are consitent and straight forward:

**f**: objective function in the form of f(x, *args)

**x**: Root guess

**x0, x1**: Root bracket

**xtol=1e-8**: Solver stops when the root lies within `xtol`.

**ytol=5e-8**: Solver stops when the f(x) lies within `ytol` of the root.

**yval=0**: Root offset. Solver will find x where f(x) = `yval`.

**args=()**: Arguments to pass to `f`.

Flexsolve includes the following solvers:

* For solving f(x) = x:
  * fixed_point: Simple fixed point iteration.
  * wegstein: Wegstein's acceleration method.
  * aitken: Aitken-Steffensen acceleration method.
* For solving f(x) = yval and x0 < x < x1:
  * bisection: Simple bisection method
  * false_position: Simple false position method.
  * IQ_interpolation: Inter-quadratic interpolation (similar to `scipy.optimize.brentq <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.brentq.html>`__)
  * bounded_wegstein: False position method with Wegstein acceleration.
  * bounded_aitken: False position method with Aitken-Steffensen acceleration.
* For solving f(x) = 0:
  * secant: Simple secant method.
  * wegstein_secant: Secant method with Wegstein acceleration.
  * aitken_secant: Secant method with Aitken acceleration.

Here are some exmples using flexsolve's Profiler object to test and compare
different solvers.:

.. code-block:: python

    >>> import flexsolve as flx 
    >>> from scipy import optimize as opt
    >>> x0, x1 = [-5, 5]
    >>> f = lambda x: x**3 - 40 + 2*x 
    >>> p = flx.Profiler(f)
    >>> opt.brentq(p, x0, x1, xtol=1e-8)
    3.225240462778411
    >>> p.archive('[Scipy] Brent-Q') # Save/archive results with given name
    >>> opt.brenth(p, x0, x1)
    3.2252404627917794
    >>> p.archive('[Scipy] Brent-H')
    >>> flx.IQ_interpolation(p, x0, x1)
    3.225240462796626
    >>> p.archive('IQ-interpolation')
    >>> flx.bounded_wegstein(p, x0, x1)
    3.225240462790051
    >>> p.archive('Wegstein')
    >>> x_aitken = flx.bounded_aitken(p, x0, x1)
    3.2252404627883218
    >>> p.archive('Aitken')
    >>> p.plot(r'$f(x) = 0 = x^3 + 2 \cdot x - 40$ where $-5 < x < 5$')

.. image:: https://raw.githubusercontent.com/yoelcortes/flexsolve/master/docs/images/bounded_solvers_example.png

.. code-block:: python

    >>> p = flx.Profiler(f)
    >>> x_guess = -5
    >>> flx.wegstein_secant(p, x_guess)
    3.22524046279178
    >>> p.archive('Wegstein')
    >>> flx.aitken_secant(p, x_guess)
    3.22524046279178
    >>> p.archive('Aitken')
    >>> flx.secant(p, x_guess)
    3.2252404627918057
    >>> p.archive('Secant')
    >>> opt.newton(p, x_guess)
    3.2252404627918065
    >>> p.archive('[Scipy] Newton')
    >>> p.plot(r'$f(x) = 0 = x^3 + 2 \cdot x - 40$')

.. image:: https://raw.githubusercontent.com/yoelcortes/flexsolve/master/docs/images/fixed_point_solvers_example.png

.. code-block:: python

    >>> # Note that x = 40/x^2 - 2/x is the same
    >>> # objective function as x**3 - 40 + 2*x = 0
    >>> f = lambda x: 40/x**2 - 2/x
    >>> p = flx.Profiler(f)
    >>> x_guess = 5.
    >>> flx.wegstein(p, x_guess)
    3.2252404626726996
    >>> p.archive('Wegstein')
    >>> flx.aitken(p, x_guess)
    3.2252404627250075
    >>> p.archive('Aitken')
    >>> p.plot(r'$f(x) = x = \frac{40}{x^2} - \frac{2}{x}$',
    ...        markbounds=False)
    >>> # Fixed iteration is non-convergent for this equation,
    >>> # so we do not include it here

.. image:: https://raw.githubusercontent.com/yoelcortes/flexsolve/master/docs/images/general_solvers_example.png

Bug reports
-----------

To report bugs, please use the eqsolvers's Bug Tracker at:

    https://github.com/yoelcortes/flexsolve


License information
-------------------

See ``LICENSE.txt`` for information on the terms & conditions for usage
of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the eqsolvers license, if it is convenient for you,
please cite eqsolvers if used in your work. Please also consider contributing
any changes you make back, and benefit the community.


Citation
--------

To cite eqsolvers in publications use:

    Yoel Cortes-Pena (2019). flexsolve: Flexible function solvers.
    https://github.com/yoelcortes/flexsolve
