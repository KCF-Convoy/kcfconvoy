#!/bin/env python
# coding: utf-8

import numpy as np


def similarity(kcf_vec_1, kcf_vec_2, n_nodes=list(range(99)),
               levels=[0, 1, 2]):
    kegg_atom_levels = ["atom_species", "atom_class", "kegg_atom"]
    kegg_atom_levels = set([kegg_atom_levels[level] for level in levels])
    l_count_1 = []
    l_count_2 = []
    for ele, label in kcf_vec_1.kegg_atom_label.items():
        if not label["n_nodes"] in n_nodes:
            continue
        if not label["ele_level"] in kegg_atom_levels:
            continue
        if ele in kcf_vec_2.kegg_atom_label.keys():
            l_count_1.append(kcf_vec_1.kegg_atom_label[ele]["count"])
            l_count_2.append(kcf_vec_2.kegg_atom_label[ele]["count"])
        else:
            l_count_1.append(kcf_vec_1.kegg_atom_label[ele]["count"])
            l_count_2.append(0)
    for ele, label in kcf_vec_2.kegg_atom_label.items():
        if not label["n_nodes"] in n_nodes:
            continue
        if not label["ele_level"] in kegg_atom_levels:
            continue
        if ele not in kcf_vec_1.kegg_atom_label.keys():
            l_count_1.append(0)
            l_count_2.append(kcf_vec_2.kegg_atom_label[ele]["count"])

    np_count_1 = np.array(l_count_1)
    np_count_2 = np.array(l_count_2)

    # 谷本係数の計算に必要な成分
    only_1 = np.sum(np.fmax(0, (np_count_1 - np_count_2)))
    only_2 = np.sum(np.fmax(0, (np_count_2 - np_count_1)))
    both_12 = np.sum((np.minimum(np_count_1, np_count_2)))

    # 重みつき谷本係数
    if only_1 + only_2 + both_12 == 0:
        x = 0
    else:
        x = both_12 / (only_1 + only_2 + both_12)

    # 分子１が分子２中でどのくらい保存されているかを表す係数
    if only_1 + both_12 == 0:
        x_12 = 0
    else:
        x_12 = both_12 / (only_1 + both_12)

    # 分子２が分子１中でどのくらい保存されているかを表す係数
    if only_2 + both_12 == 0:
        x_21 = 0
    else:
        x_21 = both_12 / (only_2 + both_12)

    return (x, x_12, x_21)
