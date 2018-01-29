# at kcfconvoy
import numpy as np


def similarity(v1, v2, n_nodes=range(99), levels=[0, 1, 2]):
    keggatom_levels = ['atom_species', 'atom_class', 'kegg_atom']
    a1 = []
    a2 = []
    for i, s in enumerate(v1.strs):
        if not v1.n_nodes[i] in n_nodes:
            continue
        if not v1.levels[i] in [keggatom_levels[lev] for lev in levels]:
            continue
        if s in v2.strs:
            j = v2.strs.index(s)
            a1.append(v1.counts[i])
            a2.append(v2.counts[j])
        else:
            a1.append(v1.counts[i])
            a2.append(0)
    for j, s in enumerate(v2.strs):
        if not v2.n_nodes[j] in n_nodes:
            continue
        if not v1.levels[i] in [keggatom_levels[lev] for lev in levels]:
            continue
        if s in v1.strs:
            pass
        else:
            a1.append(0)
            a2.append(v2.counts[j])

    f1 = np.array(a1)
    f2 = np.array(a2)

    # 谷本係数の計算に必要な成分
    only1 = sum([i if i > 0 else 0 for i in (f1 - f2)])
    only2 = sum([i if i > 0 else 0 for i in (f2 - f1)])
    both12 = sum([i if i < j else j for i, j in zip(f1, f2)])

    # 重みつき谷本係数
    x = both12 / (only1 + only2 + both12)
    if (only1 + only2 + both12) == 0:
        x = 0

    # 分子１が分子２中でどのくらい保存されているかを表す係数
    x12 = both12 / (only1 + both12)
    if (only1 + both12) == 0:
        x12 = 0

    # 分子２が分子１中でどのくらい保存されているかを表す係数
    x21 = both12 / (only2 + both12)
    if (only2 + both12) == 0:
        x21 = 0

    return (x, x12, x21)
