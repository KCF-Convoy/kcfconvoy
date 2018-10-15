# coding: utf-8

import numpy as np
import re
#from sklearn.cross_validation import train_test_split # 訓練データとテストデータに分割
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix # 混合行列
from sklearn.decomposition import PCA #主成分分析
from sklearn.linear_model import LogisticRegression # ロジスティック回帰
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier # K近傍法
from sklearn.svm import SVC # サポートベクターマシン
from sklearn.tree import DecisionTreeClassifier # 決定木
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier # AdaBoost
from sklearn.naive_bayes import GaussianNB # ナイーブ・ベイズ
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # Linear Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA # Quadratic Discriminant Analysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

def similarity(kcf_vec_1, kcf_vec_2, n_nodes=list(range(99)),
               levels=[0, 1, 2]):
    kegg_atom_levels = ["atom_species", "atom_class", "kegg_atom"]
    kegg_atom_levels = set([kegg_atom_levels[level] for level in levels])
    l_count_1 = []
    l_count_2 = []
    for ele, label in kcf_vec_1.kcf_vec.items():
        if not label["n_nodes"] in n_nodes:
            continue
        if not label["ele_level"] in kegg_atom_levels:
            continue
        if ele in kcf_vec_2.kcf_vec.keys():
            l_count_1.append(kcf_vec_1.kcf_vec[ele]["count"])
            l_count_2.append(kcf_vec_2.kcf_vec[ele]["count"])
        else:
            l_count_1.append(kcf_vec_1.kcf_vec[ele]["count"])
            l_count_2.append(0)
    for ele, label in kcf_vec_2.kcf_vec.items():
        if not label["n_nodes"] in n_nodes:
            continue
        if not label["ele_level"] in kegg_atom_levels:
            continue
        if ele not in kcf_vec_1.kcf_vec.keys():
            l_count_1.append(0)
            l_count_2.append(kcf_vec_2.kcf_vec[ele]["count"])

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

class Classifiers:
    def __init__(self):
        self.classifiers = [
            ["Random Forest", RandomForestClassifier()],
            ["Logistic Regression", LogisticRegression()],
            ["Stochastic Gradient Descent", SGDClassifier()],
            ["Nearest Neighbors", KNeighborsClassifier()],
            ["Linear SVM", SVC(kernel="linear")],
            ["Polynomial SVM", SVC(kernel="poly")],
            ["RBF SVM", SVC(kernel="rbf")],
            ["Sigmoid SVM", SVC(kernel="sigmoid")],
            ["Decision Tree", DecisionTreeClassifier()],
            ["Extra Tree", ExtraTreesClassifier()],
            ["Gradient Boosting", GradientBoostingClassifier()],
            ["AdaBoost", AdaBoostClassifier()],
            ["Naive Bayes", GaussianNB()],
            ["Linear Discriminant Analysis", LDA()],
            ["Quadratic Discriminant Analysis", QDA()],
            ["Gaussian Process", GaussianProcessClassifier()],
            ["Multi-Layer Perceptron", MLPClassifier()]
        ]