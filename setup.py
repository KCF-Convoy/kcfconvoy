#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages
from kcfconvoy import __author__, __version__, __license__

setup(
        name             = 'kcfconvoy',
        version          = __version__,
        description      = 'Sample for installing python library from github using pip',
        license          = __license__,
        author           = __author__,
        author_email     = 'maskot1977@gmail.com',
        url              = 'https://github.com/KCF-Convoy',
        keywords         = 'sample pip github python',
        packages         = find_packages(),
        install_requires = ["matplotlib",
                            "networkx",
                            "numpy",
                            "IPython",
                            "pandas",
                            "pillow"
                            ],
        )
