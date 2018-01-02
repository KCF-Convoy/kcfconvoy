#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages
from kcfconvoy import __author__, __version__, __license__

setup(
        name             = 'kcfconvoy',
        version          = __version__,
        url              = 'https://github.com/KCF-Convoy',
        author           = __author__,
        author_email     = 'maskot1977@gmail.com',
        description      = 'KCF convoy: requires rdkit installed in advance',
        license          = __license__,
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
