# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:17:00 2017

@author: Yoel Cortes-Pena
"""
from setuptools import setup

setup(
    name='flexsolve',
    packages=['flexsolve'],
    license='MIT',
    version='0.3.7',
    description='Flexible function solvers',
    long_description=open('README.rst').read(),
    author='Yoel Cortes-Pena',
    install_requires=['numba>=0.48.0', 'llvmlite>=0.31', 'numpy'],
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
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Mathematics'],
    keywords='solve equation function flexible',
)