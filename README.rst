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

    import flexsolve as flx 
    from scipy import optimize as opt
    
    x0, x1 = [-5, 5]
    f = lambda x: x**3 - 40 + 2*x 
    p = flx.Profiler(f)
    x_brentq = opt.brentq(p, x0, x1, xtol=1e-8)
    p.archive('[Scipy] Brent-Q') # Save/archive results with given name
    x_brenth = opt.brenth(p, x0, x1)
    p.archive('[Scipy] Brent-H')
    x_IQ = flx.IQ_interpolation(p, x0, x1)
    p.archive('IQ-interpolation')
    x_wegstein = flx.bounded_wegstein(p, x0, x1)
    p.archive('Wegstein')
    x_aitken = flx.bounded_aitken(p, x0, x1)
    p.archive('Aitken')
    p.plot()

.. image:: https://raw.githubusercontent.com/yoelcortes/flexsolve/tree/master/docs/images/bounded_solvers_example.png

.. code-block:: python

    p = flx.Profiler(f)
    x_guess = -5
    x_wegstein_secant = flx.wegstein_secant(p, x_guess)
    p.archive('Wegstein')
    x_aitken_secant = flx.aitken_secant(p, x_guess)
    p.archive('Aitken')
    x_secant = flx.secant(p, x_guess)
    p.archive('Secant')
    x_newton = opt.newton(p, x_guess)
    p.archive('[Scipy] Newton')
    p.plot()

.. image:: https://raw.githubusercontent.com/yoelcortes/flexsolve/tree/master/docs/images/general_solvers_example.png

.. code-block:: python

    # Note that x = 40/x^2 - 2/x is the same
    # objective function as x**3 - 40 + 2*x = 0
    f = lambda x: 40/x**2 - 2/x
    p = flx.Profiler(f)
    x_guess = 5.
    x_wegstein = flx.wegstein(p, x_guess)
    p.archive('Wegstein')
    x_aitken = flx.aitken(p, x_guess)
    p.archive('Aitken')
    p.plot(markbounds=False)
    # Fixed iteration is non-convergent for this equation,
    # so we do not include it here

.. image:: https://raw.githubusercontent.com/yoelcortes/flexsolve/tree/master/docs/images/fixed_point_solvers_example.png

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
