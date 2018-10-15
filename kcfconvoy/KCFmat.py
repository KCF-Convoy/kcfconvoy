# coding: utf-8

import numpy as np
import re
from .KCFvec import KCFvec
from .Library import Library
from sklearn.model_selection import train_test_split
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

class KCFmat(Library):
    def __init__(self):
        super().__init__()
        self.kcf_vecs = []
        self.all_strs = []
        self.all_mat = np.array([])
        self.mask_array = np.array([])
        self.mat = np.array([])
        self.selected_features = []
        self.brite_group = {}
        self.strs = []

    def input_from_kegg(self, cid, name=None):
        if super().input_from_kegg(cid, name):
            self._append_kcf_vec()
        else:
            return False

        return True

    def input_from_knapsack(self, cid, name=None):
        if super().input_from_knapsack(cid, name):
            self._append_kcf_vec()
        else:
            return False

        return True

    def input_molfile(self, molfile, name=None):
        if super().input_molfile(molfile, name):
            self._append_kcf_vec()
        else:
            return False

        return True

    def input_inchi(self, inchi, name=None):
        if super().input_inchi(inchi, name):
            self._append_kcf_vec()
        else:
            return False

        return True

    def input_smiles(self, smiles, name=None):
        if super().input_smiles(smiles, name):
            self._append_kcf_vec()
        else:
            return False

        return True

    def input_rdkmol(self, mol, name=None):
        if super().input_rdkmol(mol, name):
            self._append_kcf_vec()
        else:
            return False

        return True

    def _append_kcf_vec(self):
        kcf_vec = KCFvec()
        kcf_vec.input_rdkmol(self.cpds[-1].mol, self.names[-1])
        kcf_vec.convert_kcf_vec()
        self.kcf_vecs.append(kcf_vec)
        self.all_strs = \
            list(set(self.all_strs) | kcf_vec.kcf_vec.keys())

        return True

    def _get_num_string(self, kcfvec, kcfstring):
        return len(kcfvec.string2seq(kcfstring))
        #if string in kcfvec.string2seq().keys():
        #    return len(kcfvec.string2seq()[string])
        #else:
        #    return 0   

    def calc_kcf_matrix(self, ratio=400):
        self.all_mat = np.array([[self._get_num_string(kcfvec, string) for string in self.all_strs] for kcfvec in self.kcf_vecs])
        kcf_mat_T = self.all_mat.T
        min_cpd = max(len(self.all_mat) / ratio, 1)
        self.mask_array = np.array(
            [len(np.where(a != 0)[0]) > min_cpd for a in kcf_mat_T])
        kcf_mat_T_2 = kcf_mat_T[self.mask_array]
        self.mat = kcf_mat_T_2.T
        self.strs = [self.all_strs[idx] for (idx, b) in enumerate(self.mask_array) if b]

        return True
    
    def feature_selection(self, y, classifier, num_trial = 10, corrcoef_threshold = 0.8, max_features = 2048):
        X = self.mat
        accum_feature_importances_ = np.zeros(len(X[0]))
        for trial in range(num_trial):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4) 
            clf = classifier
            clf.fit(X_train, y_train)
            accum_feature_importances_ += clf.feature_importances_
            
        ranking_feature_importances_ = np.argsort(accum_feature_importances_)[::-1]
        for feature_idx in ranking_feature_importances_:
            if len(self.selected_features) == 0:
                self.selected_features.append(feature_idx)
            else:
                selected_flag = True
                for feature_idx2 in self.selected_features:
                    corrcoef = np.corrcoef(self.mat.T[feature_idx], self.mat.T[feature_idx2])[0][1]
                    if corrcoef > corrcoef_threshold:
                        selected_flag = False
                        break
                if selected_flag:
                    self.selected_features.append(feature_idx)
            if len(self.selected_features) == max_features:
                break
                
        return True

    def selected_mat(self):
        return self.mat[:, self.selected_features]

    def input_from_brite(self, brite_file):
        alphabet = list('ABCDEFGHIJKLKMOPQRSTUVWXYZ')
        hierarchy = {}
        _id = ''
        with open(brite_file) as f:
            for line in f.readlines():
                head = line[0:1]
                if head in alphabet:
                    if head not in hierarchy.keys():
                        hierarchy[head] = ''
                    if re.search(r'[CD]\d{5}', line):
                        _id = line.split()[1]
                        if len(_id) == 0:
                            continue
                        self.input_from_kegg(_id, name=_id)
                    else:
                        hierarchy[head] = line[1:].strip()
                    for level, name in hierarchy.items():
                        key = level + ":" + name
                        if key not in self.brite_group.keys():
                            self.brite_group[key] = []
                        self.brite_group[key].append(_id)
        return True 
    
    def fps_mat(self):
        return [[int(fp.ToBitString()[i:i+1]) for i in range(len(fp.ToBitString()))] for fp in self.fps]

    def list_brite_groups(self):
        result = []
        group_names = sorted(list(self.brite_group.keys()))
        for i, group_name in enumerate(group_names):
            if len(self.brite_group[group_name]) < 10:
                continue
            result.append((i, group_name, len(self.brite_group[group_name])))
        return result
    
    def brite_class(self, i):
        return [1 if id in self.brite_group[self.list_brite_groups()[i][1]] else 0 for id in self.names]
    
    def draw_cpds(self, kcfstring='', kcfstringidx=False):
        if kcfstring == '':
            if kcfstringidx:
                kcfstring = self.strs[kcfstringidx]
            else:
                mols = [cpd.mol for cpd in self.cpds]
                return Draw.MolsToGridImage(mols, molsPerRow=3, useSVG=True, legends=self.names)
        highlightAtomLists = []
        highlightmols = []
        highlightnames = []
        for name, kcfvec in zip(self.names, self.kcf_vecs):
            if len(kcfvec.string2seq(kcfstring)) == 0:
                continue
            list1 = ",".join(kcfvec.string2seq(kcfstring)).split(",")
            highlightmols.append(kcfvec.mol)
            highlightAtomLists.append(tuple(
                [int(n) for n in list1 if n != '']
            ))
            highlightnames.append(name)
        return Draw.MolsToGridImage( highlightmols, molsPerRow=3, useSVG=True, legends=highlightnames, highlightAtomLists=highlightAtomLists)
 