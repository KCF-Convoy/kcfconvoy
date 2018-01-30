#!/bin/env python
# coding: utf-8

import numpy as np

from .KCFvec import KCFvec
from .Library import Library


class KCFmat(Library):
    def __init__(self):
        super().__init__()
        self.kcf_vecs = []
        self.all_strs = []
        self.all_mat = np.array([])
        self.mask_array = np.array([])
        self.mat = np.array([])

    def input_from_kegg(self, cid, name=None):
        super().input_from_kegg(cid, name)
        self._append_kcf_vec()

        return True

    def input_from_knapsack(self, cid, name=None):
        super().input_from_knapsack(cid, name)
        self._append_kcf_vec()

        return True

    def input_molfile(self, molfile, name=None):
        super().input_molfile(molfile, name)
        self._append_kcf_vec()

        return True

    def input_inchi(self, inchi, name=None):
        super().input_inchi(inchi, name)
        self._append_kcf_vec()

        return True

    def input_smiles(self, smiles, name=None):
        super().input_from_smiles(smiles, name)
        self._append_kcf_vec()

        return True

    def input_rdkmol(self, mol, name=None):
        super().input_rdkmol(mol, name)
        self._append_kcf_vec()

        return True

    def _append_kcf_vec(self):
        kcf_vec = KCFvec()
        kcf_vec.input_rdkmol(self.cpds[-1], self.names[-1])
        kcf_vec.convert_kcf_vec
        self.kcf_vecs.append(kcf_vec)
        self.all_strs = \
            list(set(self.all_strs) | kcf_vec.kegg_atom_label.keys())

        return True

    def calc_kcf_matrix(self, ratio=400):
        self.all_mat = np.zeros((len(self.kcf_vecs), len(self.all_strs)))
        for i, kcf_vec in enumerate(self.kcf_vecs):
            for ele, label in kcf_vec.items():
                self.all_mat[i][self.all_strs.index(ele)] = label["count"]
        kcf_mat_T = self.all_mat.T
        min_cpd = max(len(self.all_mat) / ratio, 1)
        self.mask_array = np.array(
            [len(np.where(a != 0)[0]) > min_cpd for a in kcf_mat_T])
        kcf_mat_T_2 = kcf_mat_T[self.mask_array]
        self.mat = kcf_mat_T_2.T

        return True
