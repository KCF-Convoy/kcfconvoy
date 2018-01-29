#!/bin/env python
# coding: utf-8

from collections import defaultdict
from copy import deepcopy

import pandas as pd

import networkx as nx
from rdkit import Chem
import numpy as np

from . import Library


class KCFmat(Library):
    def __init__(self):
        self.names = []
        self.kcfvecs = []
        self.all_strs = []
        self.all_mat = np.array([])
        self.mask_array = np.array([])
        self.mat = np.array([])

    def input_library(self, library):
        self.names = library.names
        self.kcfvecs = [kcf_vec(cpd) for cpd in library.cpds]
        self.all_strs = list(
            set([item for kcfvec in self.kcfvecs for item in kcfvec.strs]))
        self.all_mat = np.zeros((len(self.kcfvecs), len(self.all_strs)))
        for i, kcfvec in enumerate(self.kcfvecs):
            for j, _str in enumerate(kcfvec.strs):
                self.all_mat[i][self.all_strs.index(_str)] = kcfvec.counts[j]
        self.calc_kcf_mat()

    def calc_kcf_mat(self, ratio=400):
        kcf_matT = self.all_mat.T
        min_cpd = max(len(self.all_mat) / ratio, 1)
        self.mask_array = np.array(
            [len(np.where(a != 0)[0]) > min_cpd for a in kcf_matT])
        kcf_matT2 = kcf_matT[self.mask_array]
        self.mat = kcf_matT2.T
