#!/usr/bin/env python
# coding: utf-8
from IPython.utils.io import rprint
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from time import sleep

import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import urllib.request
 
__author__  = 'maskot1977'
__version__ = '0.0.1'
__license__ = 'MIT'