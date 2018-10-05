# coding: utf-8

import os
import urllib.request
from copy import copy

import matplotlib.pyplot as plt

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


class Compound:
    """
    Daylight Molfile and SDF
        - 2010
            http://infochim.u-strasbg.fr/recherche/Download/Fragmentor/MDL_SDF.pdf
        - 2003
            http://www.daylight.com/meetings/mug05/Kappler/ctfile.pdf
    RDKit
        http://www.rdkit.org/GettingStartedInPython.pdf (2012)
    NetworkX 2.0
        https://networkx.github.io/documentation/development/_downloads/networkx_reference.pdf
    NetworkX 1.9.1
        https://networkx.github.io/documentation/networkx-1.9.1/_downloads/networkx_reference.pdf
    NetworkX 1.9
        https://networkx.github.io/documentation/networkx-1.9/_downloads/networkx_reference.pdf
    """

    """
    method
        - input_from_kegg
            KEGG_IDからobjectを生成する。
            - input:
                kegg_id
        - input_from_knapsack
            knapsackからobjectを生成する。
            - input:
                knapsack_id
        - input_molfile
            molfileのpathからobjectを生成する。
            - input:
                molfile_path
        - input_inchi
            inchi形式からobjectを生成する。
            - input:
                inchi
        - input_smiles
            smiles形式からobjectを生成する。
            - input:
                smiles
        - input_rdkmol
            rdkitのmol形式からobjectを生成する。
            - input:
                rdk_mol
        - draw_cpd
            実行ディレクトリにpng_file_name.pngで生成する。rdkitのmethodで描画している。
            - input:
                png_file_name
        - draw_cpd_with_labels
            label付けをするdraw method。labelのstartや，custom_labelを指定できる。
        - find_seq
            self.graph以下に含まれるグラフ構造のうち，length 以下の部分構造をイテレーターとして返す。
            length, bidirectonal の pram を持つ。
            - input:
                nodeのlabelのlist。
        - has_bond
            atom_1 と atom_2 の間にedgeがあるか bool で返す。
            - input:
                atom_1
                atom_2
        - get_symbol
            atom の index を受け取って，atomのsymbol を返す。
            - input:
                atom_index
        - get_triplets
            3連続の結合を全て取り出す。
        - get_vicinities
            隣接している atom のindexを返す。

    内部 method:
        - _input_molblock
            input系のmethodから呼ばれ，実際にself.graphなどを生成する。
        - _compute_2d_coords
            2Dの座標が生成されていない場合は生成する。self._set_coordinates()も呼ぶ。
        - self._set_coordinates
            2Dの座標をself.graph.nodesに登録する。
        - self._get_coordinates
            2Dの座標を得る。
        - _get_node_colors
            draw_cpd_with_labels の内部メソッド
        - _get_node_labels
            draw_cpd_with_labels の内部メソッド
    """

    def __init__(self):
        """
        self.heads : molblock形式の3行目(0-indexed)
            ex.)   4  4  0  0  0  0  0  0  0  0999 V2000
        self.tail : molblock形式の最後らへんの行
        self.n_atoms : graphに含まれるatomの数
        self.n_bonds : graphに含まれるedgeの数
        self.fit2d : 2Dが計算されているかどうか，されている場合，self.graph[node]["row"]
            の最初の項目などに座標情報が入っている。
        self.graph : グラフ構造をもつ。
            node のメタデータとして，["symbol", "row"]
            edge のメタデータとして，["order", "index", "row"]
        self.mol : rdkitのmol形式として登録されている。
        """
        self.heads = None
        self.tails = []
        self.n_atoms = None
        self.n_bonds = None
        self.fit2d = False
        self.graph = nx.Graph()
        self.mol = None

    def input_from_kegg(self, cid):
        """
        e.g., cid = C00002
        KEGG_IDからmol形式でdownloadし，./kegg/cid.molとして保存する。
        self.input_molfileを呼ぶ。
        """
        kegg_dir = "kegg"
        if not os.path.isdir("./" + kegg_dir):
            os.mkdir("./" + kegg_dir)
        if not os.path.isfile("./{}/{}.mol".format(kegg_dir, cid)):
            url = "http://www.genome.jp/dbget-bin/www_bget?-f+m+{}".format(cid)
            urllib.request.urlretrieve(url,
                                       "./{}/{}.mol".format(kegg_dir, cid))

        self.input_molfile("./{}/{}.mol".format(kegg_dir, cid))

        return True

    def input_from_knapsack(self, cid):
        """
        e.g., cid = C00002657
        knapsack_IDからmol形式でdownloadし，./knapsack/cid.molとして保存する。
        self.input_molfileを呼ぶ。
        knapsack3dのmolファイルは3d座標が入っているため座標は再計算させる。
        """
        knapsack_dir = "knapsack"
        if not os.path.isdir("./" + knapsack_dir):
            os.mkdir("./" + knapsack_dir)
        if not os.path.isfile("./{}/{}.mol".format(knapsack_dir, cid)):
            url = "http://knapsack3d.sakura.ne.jp/mol3d/{}.3d.mol".format(cid)
            urllib.request.urlretrieve(url,
                                       "./{}/{}.mol".format(knapsack_dir, cid))
        mol = Chem.MolFromMolFile("./{}/{}.mol".format(knapsack_dir, cid))
        Chem.MolToMolFile(mol, "./{}/{}.mol".format(knapsack_dir, cid))

        self.input_molfile("./{}/{}.mol".format(knapsack_dir, cid))
        self.fit2d = False

        return True

    def input_molfile(self, molfile):
        """
        molfileの実体ではなく，molfileのpathを受け取る。
        molfileの中身自体はmolblock。
        self._input_molblockをすることで，諸々の情報をselfに持たせる。
        ここで，self._compute_2d_coords()を呼んでおく。
        """
        with open(molfile, "r") as f:
            molblock = f.read()

        self._input_molblock(molblock)
        self.mol = Chem.MolFromMolBlock(molblock)

        if self.fit2d is False:
            self._compute_2d_coords()

        return True

    def input_inchi(self, inchi):
        """
        inchi形式を受け取って，rdkitのmol形式に直し，mol形式をmolblockに直し登録する。
        """
        self.input_rdkmol(Chem.MolFromInchi(inchi))

        return True

    def input_smiles(self, smiles):
        """
        smiles形式を受け取って，rdkitのmol形式に直し，mol形式をmolblockに直し登録する。
        """
        self.input_rdkmol(Chem.MolFromSmiles(smiles))

        return True
    
    def input_rdkmol(self, rdkmol):
        """
        rdkitのmol形式を受け取り，mol形式をmolblockに直し登録する。
        """
        self.mol = rdkmol
        self._input_molblock(Chem.MolToMolBlock(rdkmol))

        if self.fit2d is False:
            self._compute_2d_coords()

        return True

    def _input_molblock(self, molblock):
        """
        molblockの形式として，
            0 - 2 行目は必要のない行
            3 行目はn_atomsとn_edgesが含まれている。self.headとして定義する。
            (3+1) - (3+1+n_atoms-1) 行目はatomについての情報。
                ex:) n_atoms=4とすると 4 - 7 行目を指す。
            (3+1+n_atoms) - (3+1+n_atoms+n_edges-1) 行目はedgeについての情報。
                ex.) n_atoms=4,n_edges=4とすると 8 - 11 行目を指す。
            (3+1+n_atoms+n_edges) 行目以降は "CHG" が含まれていない限り，
            self.tailにappendする。

        atomについての行の最初の項目が 0.0 出ない場合，2Dについてがすでに計算されている。

        molblock の中身の諸々については，self.head, self.tail, self.graph[node]["row"]
        のいずれかに登録される。
        """
        l_molblock = molblock.split("\n")
        # 3 行目
        line = l_molblock[3]
        self.n_atoms, self.n_bonds = \
            map(int, [line[:3].strip(), line[3:6].strip()])
        self.head = l_molblock[3]

        index_check_list = []
        # atom についての行
        for i in range(4, 4 + self.n_atoms):
            line = l_molblock[i].split()
            if line[12] == "0":
                line[12] = str(len(self.graph.nodes()) + 1)
            if float(line[0]) != 0.0:
                self.fit2d = True
            if line[3] == "H":
                continue
            index_check_list.append(i - 4)
            self.graph.add_node(
                len(self.graph.nodes()), symbol=line[3], row=line)

        # edge についての行
        for i in range(4 + self.n_atoms, 4 + self.n_atoms + self.n_bonds):
            line = l_molblock[i]
            line = list(map(int, [line[:3].strip(), line[3:6].strip(),
                                  line[6:9].strip(), line[9:12].strip()]))
            if line[2] == 2:
                line[3] = 0
            bondidx = i - 4 - self.n_atoms
            if line[0] - 1 not in index_check_list:
                continue
            if line[1] - 1 not in index_check_list:
                continue
            replaced_list = copy(line)
            replaced_list[0] = index_check_list.index(line[0] - 1) + 1
            replaced_list[1] = index_check_list.index(line[1] - 1) + 1
            self.graph.add_edge(index_check_list.index(line[0] - 1),
                                index_check_list.index(line[1] - 1),
                                order=line[2], index=bondidx,
                                row=list(map(str, replaced_list)))

        for line in l_molblock[:4 + self.n_atoms + self.n_bonds]:
            if "CHG" in line.split():
                continue
            self.tails.append(line)

        return True

    def _compute_2d_coords(self):
        """
        化合物の2D構造の生成。
        内部で self.set_coordinates を呼ぶ。
        input_from_kegg, input_from_knapsack, input_inchi, input_rdkmol,
        input_molfile それぞれをした時，self.fit2dがFalseだったら呼ばれる。
        """
        AllChem.Compute2DCoords(self.mol)
        self._set_coordinates()
        self.fit2d = True

        return True

    def _set_coordinates(self):
        """
        AllChem.Compute2DCoords(self.mol) によって，mol形式のなかで2D座標が計算される。
        それを，molblock形式に戻した後，self.graphのそれぞれに登録していく。
        前提として，self._input_molblockが呼ばれているため，self.graph[node]["row"]
        などは存在している。
        """
        molblock = Chem.MolToMolBlock(self.mol)
        l_molblock = molblock.split("\n")

        for i in range(4, 4 + self.n_atoms):
            line = l_molblock[i].split()
            if i - 4 in self.graph.node.keys():
                self.graph.node[i - 4]["row"][0] = line[0]
                self.graph.node[i - 4]["row"][1] = line[1]
                self.graph.node[i - 4]["row"][2] = line[2]

        return True

    def _get_coordinates(self):
        """
        self.graph.nodesのrow属性にそれぞれmolblockの行が含まれている。
        その行のうち，0, 1番目に二次元座標が登録されている。
        それらの二次元座標をネストしたlistとして返す。
        """
        l_rows = [node[1]["row"] for node in self.graph.nodes(data=True)]
        l_coordinates = [list(map(float, row[0:2])) for row in l_rows]

        return l_coordinates

    def draw_cpd(self, image_file="mol.png"):
        """
        化合物の描画。
        2D構造が生成されていない場合，生成する。
        image_file の名前で実行ディレクトリに保存する。
        """
        if not self.fit2d:
            self._compute_2d_coords()

        Draw.MolToFile(self.mol, image_file)

        return True

    def draw_cpd_with_labels(self, start=0, custom_label=None):
        """
        ラベル付きの化合物の描画。
        2D構造が生成されていない場合，生成する。
        """
        if not self.fit2d:
            self._compute_2d_coords()

        pos = self._get_coordinates()
        node_color = self._get_node_colors()
        if custom_label is None:
            node_label = self._get_node_labels(start)
        else:
            node_label = custom_label
        edge_labels = dict()
        for edge in self.graph.edges(data=True):
            edge_labels[(edge[0], edge[1])] = edge[2]["index"]
        nx.draw_networkx(self.graph, pos, node_color=node_color, alpha=0.4)
        nx.draw_networkx_labels(self.graph, pos, fontsize=6, labels=node_label)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels,
                                     fontsize=3, font_color="b")
        plt.draw()

        return True

    def _get_node_colors(self):
        """
        それぞれのnodesのsymbolのlistから，
        listをsortして，重複無しで最初から0-indexのintを振り，そのintとsymbol_listとの対応を返す。
        """
        symbol_list = \
            [node[1]["symbol"] for node in self.graph.nodes(data=True)]

        count = 0
        d_index = dict()
        for symbol in sorted(symbol_list):
            if symbol in d_index.keys():
                continue
            else:
                d_index[symbol] = count
                count += 1

        l_index = [d_index[symbol] for symbol in symbol_list]

        return l_index

    def _get_node_labels(self, start=0):
        """
        nodeのlabelとなるdictを返す。
        networkxの関係上，dictである必要がある。
        基本的に，0-indexか1-indexかを調整するためのもの
        """
        d_label = {}
        l_nodes = self.graph.nodes(data=True)
        for node in l_nodes:
            d_label[node[0]] = str(int(node[1]["row"][12]) + start - 1)

        return d_label

    def find_seq(self, length, bidirectonal=True):
        """
        self.graph以下に含まれるグラフ構造のうち，length 以下の部分構造を
        イテレーターとして返す。
        返される型，nodeのlabelのlist。
        """
        for i in self.graph.nodes():
            for j in self.graph.nodes():
                if length == 3 or i == j:
                    continue
                if bidirectonal:
                    pass
                elif i >= j:
                     continue
                for seq in nx.all_simple_paths(self.graph, i, j, cutoff=(length)):
                    if len(seq) == 2:
                        pass
                    elif len(seq) <= length:
                        yield seq
                    if seq[0] in self.graph.adj[seq[-1]].keys():
                        seq2 = [n for n in seq]
                        seq2.append(seq[0])
                        yield seq2
                    continue
                            

    def has_bond(self, atom_1, atom_2):
        """
        TODO 入力の型
        atom_1 と atom_2 の間にedgeがあるか bool で返す。
        """
        return atom_1 in self.graph.adj[atom_2].keys()

    def get_symbol(self, atom_index):
        """
        atom の index を受け取って，atomのsymbol を返す。
        """
        return self.graph.node[atom_index]['symbol']

    def get_triplets(self):
        """
        3連続の結合を全て取り出す。
        """
        return self.find_seq(3, bidirectonal=False)

    def get_vicinities(self):
        """
        隣接している atom のindexを返す。
        [(0, [1, 2, 3]),
        (1, [3, 4, 5]),
        (2, [3, 4, 5]),
        ...
        ]
        のような形で返す。
        """
        return [(j, [i for i in nx.all_neighbors(self.graph, j)])
                for j in self.graph.nodes()]
