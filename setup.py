# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:17:00 2017

@author: Yoel Cortes-Pena
"""
from distutils.core import setup

setup(
    name='flexsolve',
    packages=['flexsolve'],
    license='MIT',
    version='0.4.6',
    description='Flexible function solvers',
    long_description=open('README.rst').read(),
    author='Yoel Cortes-Pena',
    install_requires=['numba>=0.50.0', 'llvmlite>=0.31', 'numpy'],
    python_requires=">=3.6",
    package_data=
        {'flexsolve': []},
    platforms=['Windows', 'Mac', 'Linux'],
    author_email='yoelcortes@gmail.com',
    url='https://github.com/yoelcortes/flexsolve',
    download_url='https://github.com/yoelcortes/flexsolve.git',
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Education',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Manufacturing',
                 'Intended Audience :: Science/Research',
                 'Natural Language :: English',
                 'Operating System :: MacOS',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: POSIX :: BSD',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: Unix',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: Implementation :: CPython'],
    keywords=['solve', 'equation', 'function', 'flexible'],
)