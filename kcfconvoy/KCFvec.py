#!/bin/env python
# coding: utf-8
from collections import defaultdict
from copy import deepcopy

import pandas as pd

import networkx as nx
from rdkit import Chem

from .Compound import Compound


class KCFvec(Compound):
    """
    method:
        - input_from_kegg
        - input_from_knapsack
        - input_molfile
        - input_inchi
        - input_smiles
        - input_rdkmol
        - convert_kcf_vec
            should be called after input_* methods
        - get_pandas_df
        - string2seq
    internal method:
        - _rdkmol_to_kcf
            input 系の method で rdkit の mol が登録された後に，呼ばれる。
            self.kcf の登録。
            内部で使用する method は下記の通り。
            - _set_molblock
            - _set_kegg_atoms_labels
            - _get_label_C
            - _get_label_N
            - _get_label_O
            - _get_label_S
            - _get_label_P
            - _get_label_else
            - _check_in_ring
            - _create_kegg_atom_label

        - _get_pin_path
        - _add_vec_element
        - _bund_pin_path
    """

    def __init__(self):
        """
        Compound クラスにおいて，
            - self.heads = None
            - self.tails = []
            - self.n_atoms = None
            - self.n_bonds = None
            - self.fit2d = False
            - self.graph = nx.Graph()
            - self.mol = None
        が生成されている。

        self.kcf : str 型で kcf 形式をもつ。
        self.kegg_atom_label : dict 型。rdkit の mol 形式から生成。kcf を作成する際に必要。
        self.kcf_vec : defaultdict
            key : str
            value : dict
                key : n_nodes, ele_type, ele_level, count
        """
        super().__init__()
        self.kcf = ""
        self.cpd_name = None
        self.kegg_atom_label = dict()
        self.molblock_atoms = dict()
        self.molblock_bonds = []

        self.kcf_vec = defaultdict(dict)
        self.ring_string = []
        self.subs_string = []
        self._string_2_seq = {}

    def input_from_kegg(self, cid, cpd_name="NoName"):
        """
        e.g., cid = C00002
        KEGG_IDからmol形式でdownloadし，./kegg/cid.molとして保存する。
        self.kcf の生成まで行う。
        """
        if super().input_from_kegg(cid):
            self.cpd_name = cpd_name
            self._rdkmol_to_kcf()
        else:
            return False

        return True

    def input_from_knapsack(self, cid, cpd_name="NoName"):
        """
        e.g., cid = C00002657
        knapsack_IDからmol形式でdownloadし，./knapsack/cid.molとして保存する。
        self.kcf の生成まで行う。
        """
        if super().input_from_knapsack(cid):
            self.cpd_name = cpd_name
            self._rdkmol_to_kcf()
        else:
            return False

        return True

    def input_molfile(self, molfile, cpd_name="NoName"):
        """
        molfileの実体ではなく，molfileのpathを受け取る。
        molfileの中身自体はmolblock。
        self._input_molblockをすることで，諸々の情報をselfに持たせる。
        ここで，self._compute_2d_coords()を呼んでおく。
        self.kcf の生成まで行う。
        """
        if super().input_molfile(molfile):
            self.cpd_name = cpd_name
            self._rdkmol_to_kcf()
        else:
            return False

        return True

    def input_inchi(self, inchi, cpd_name="NoName"):
        """
        inchi形式を受け取って，rdkitのmol形式に直し，mol形式をmolblockに直し登録する。
        """
        if super().input_inchi(inchi):
            self.cpd_name = cpd_name
            self._rdkmol_to_kcf()
        else:
            return False

        return True

    def input_smiles(self, smiles, cpd_name="NoName"):
        """
        smiles形式を受け取って，rdkitのmol形式に直し，mol形式をmolblockに直し登録する。
        """
        if super().input_smiles(smiles):
            self.cpd_name = cpd_name
            self._rdkmol_to_kcf()
        else:
            return False

        return True

    def input_rdkmol(self, rdkmol, cpd_name="NoName"):
        """
        rdkitのmol形式を受け取り，mol形式をmolblockに直し登録する。
        """
        if super().input_rdkmol(rdkmol):
            self.cpd_name = cpd_name
            self._rdkmol_to_kcf()
        else:
            return False

        return True

    def _rdkmol_to_kcf(self):
        """
        rdkit の mol object から kcf を生成する。
        """
        if self.kcf != "":
            self.kcf = ""
        self._set_molblock()
        self._set_kegg_atoms_labels()

        self.kcf += "ENTRY       {:<30}Compound\n".format(self.cpd_name)
        self.kcf += "ATOM        {:<5}\n".format(self.mol.GetNumAtoms())
        for index, molblock in \
                sorted(list(self.molblock_atoms.items()), key=lambda x: x[0]):
            self.kcf += " " * 12
            self.kcf += "{:<4}{:<4}{:<5}{:<9}{:<8}\n".format(
                index + 1,
                self.kegg_atom_label[index]["kegg_atom"],
                self.kegg_atom_label[index]["atom_species"],
                molblock[0], molblock[1])

        self.kcf += "BOND        {:<5}\n".format(self.mol.GetNumBonds())
        for index, molblock in enumerate(self.molblock_bonds):
            self.kcf += " " * 12
            self.kcf += "{:<3}{:>4}{:>4}{:>2}\n".format(index + 1, molblock[0],
                                                        molblock[1],
                                                        molblock[2])
        self.kcf += "///\n"

        return True

    def _set_molblock(self):
        """
        self.mol から生成した moblock 形式において
        atom の部分を dict 型の self.molblock_atoms
        edge の部分を list 型の self.molblock_bonds
        に登録する。
        """
        self.molblock_atoms = dict()
        self.molblock_bonds = []
        molblock = Chem.MolToMolBlock(self.mol)
        l_molblock = molblock.split("\n")
        # atom についての行
        for i in range(4, 4 + self.n_atoms):
            line = l_molblock[i].split()
            self.molblock_atoms[i - 4] = line[0:3]
        # edge についての行
        for i in range(4 + self.n_atoms, 4 + self.n_atoms + self.n_bonds):
            line = l_molblock[i].split()
            self.molblock_bonds.append(line)

        return True

    def _set_kegg_atoms_labels(self):
        """
        rdk の mol object から kcf 用の labels を生成する。
        self.kegg_atom_label に登録していく。
            key: atom_index
            value: kegg_atom についての dict
            となる dict である。
        rdkmol.GetAtoms は RDKit の mol object が持つ method
        atom.GetSymbol() は RDKit の atom object が持つ method
        pram data が True の場合は label 内の全ての属性を登録する。
        """
        if self.mol is None:
            return False   # TODO error処理

        for atom in self.mol.GetAtoms():
            if atom.GetSymbol() == "C":
                label = self._get_label_C(atom)
            elif atom.GetSymbol() == "N":
                label = self._get_label_N(atom)
            elif atom.GetSymbol() == "O":
                label = self._get_label_O(atom)
            elif atom.GetSymbol() == "S":
                label = self._get_label_S(atom)
            elif atom.GetSymbol() == "P":
                label = self._get_label_P(atom)
            else:
                label = self._get_label_else(atom)

            self.kegg_atom_label[atom.GetIdx()] = label

        return True

    def _get_label_C(self, atom):
        """
        引数の atom は rdkit の mol object。
        kegg_atom_label を返す。
        """
        Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
        if atom.GetIsAromatic():
            if Explicit == 3:
                kegg_atom = "C8x"
            elif Explicit == 4:
                kegg_atom = "C8y"
            else:
                kegg_atom = "C0"

        else:
            if Explicit - len(atom.GetNeighbors()) == 0:
                if self._check_in_ring(atom):
                    if Explicit == 2:
                        kegg_atom = "C1x"
                    elif Explicit == 3:
                        kegg_atom = "C1y"
                    elif Explicit == 4:
                        kegg_atom = "C1z"
                    else:
                        kegg_atom = "C0"
                else:
                    if Explicit == 1:
                        kegg_atom = "C1a"
                    elif Explicit == 2:
                        kegg_atom = "C1b"
                    elif Explicit == 3:
                        kegg_atom = "C1c"
                    elif Explicit == 4:
                        kegg_atom = "C1d"
                    else:
                        kegg_atom = "C0"

            elif Explicit - len(atom.GetNeighbors()) == 1:
                num_neighbor_oxygen_atoms = 0
                is_carbonyl = False
                is_ester = False
                for bond in atom.GetBonds():
                    if bond.GetEndAtom().GetSymbol() != "C":
                        bond_partner = bond.GetEndAtom()
                    else:
                        bond_partner = bond.GetBeginAtom()
                    if bond_partner.GetSymbol() == "O":
                        num_neighbor_oxygen_atoms += 1
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            is_carbonyl = True
                        elif bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                            if len(bond_partner.GetNeighbors()) == 2:
                                is_ester = True
                if is_carbonyl:
                    if num_neighbor_oxygen_atoms == 1:
                        if Explicit == 2:
                            kegg_atom = "C2a"
                        elif Explicit == 3:
                            kegg_atom = "C4a"
                        elif Explicit == 4:
                            if self._check_in_ring(atom):
                                kegg_atom = "C5x"
                            else:
                                kegg_atom = "C5a"
                        else:
                            kegg_atom = "C0"
                    elif num_neighbor_oxygen_atoms in [2, 3]:
                        if is_ester:
                            if self._check_in_ring(atom):
                                kegg_atom = "C7x"
                            else:
                                kegg_atom = "C7a"
                        else:
                            kegg_atom = "C6a"
                    else:
                        kegg_atom = "C0"
                else:
                    if self._check_in_ring(atom):
                        if Explicit == 3:
                            kegg_atom = "C2x"
                        elif Explicit == 4:
                            kegg_atom = "C2y"
                        else:
                            kegg_atom = "C0"
                    else:
                        if Explicit == 2:
                            kegg_atom = "C2a"
                        elif Explicit == 3:
                            kegg_atom = "C2b"
                        elif Explicit == 4:
                            kegg_atom = "C2c"
                        else:
                            kegg_atom = "C0"

            elif Explicit - len(atom.GetNeighbors()) == 2:
                if Explicit == 3:
                    kegg_atom = "C3a"
                elif Explicit == 4:
                    num_double = 0
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            num_double += 1
                    if num_double == 2:
                        kegg_atom = "C0"
                    else:
                        if self._check_in_ring(atom):
                            kegg_atom = "C0"
                        else:
                            kegg_atom = "C3b"
                else:
                    kegg_atom = "C0"

            else:
                kegg_atom = "C0"

        kegg_atom_label = self._create_kegg_atom_label(kegg_atom)

        return kegg_atom_label

    def _get_label_N(self, atom):
        Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
        if atom.GetIsAromatic():
            if len(atom.GetNeighbors()) == 2:
                if Explicit == 2:
                    kegg_atom = "N4x"
                elif Explicit == 3:
                    kegg_atom = "N5x"
                else:
                    kegg_atom = "N0"
            elif len(atom.GetNeighbors()) == 3:
                if Explicit == 3:
                    kegg_atom = "N4y"
                elif Explicit == 4:
                    kegg_atom = "N5y"
                else:
                    kegg_atom = "N0"
            else:
                kegg_atom = "N0"

        else:
            if Explicit - len(atom.GetNeighbors()) == 0:
                if self._check_in_ring(atom):
                    if Explicit == 2:
                        kegg_atom = "N1x"
                    elif Explicit == 3:
                        kegg_atom = "N1y"
                    elif Explicit == 4:     # for C06163 and others
                        kegg_atom = "N2y"
                    else:
                        kegg_atom = "N0"
                else:
                    if Explicit == 1:
                        kegg_atom = "N1a"
                    elif Explicit == 2:
                        kegg_atom = "N1b"
                    elif Explicit == 3:
                        kegg_atom = "N1c"
                    elif Explicit == 4:
                        kegg_atom = "N1d"
                    else:
                        kegg_atom = "N0"

            elif Explicit - len(atom.GetNeighbors()) == 1:
                if self._check_in_ring(atom):
                    if Explicit == 3:
                        kegg_atom = "N2x"
                    elif Explicit == 4:
                        kegg_atom = "N2y"
                    else:
                        kegg_atom = "N0"
                else:
                    if Explicit == 2:   # for C11282
                        kegg_atom = "N2a"
                    elif Explicit == 3:
                        kegg_atom = "N2b"
                    elif Explicit <= 5:     # for Nitrate
                        kegg_atom = "N2b"
                    else:
                        kegg_atom = "N0"

            elif Explicit - len(atom.GetNeighbors()) == 2:
                if len(atom.GetNeighbors()) == 1:
                    kegg_atom = "N3a"
                else:
                    if atom.GetFormalCharge() == 1:
                        nitrous = True
                        for bond in atom.GetBonds():
                            if bond.GetBondType() == \
                                    Chem.rdchem.BondType.TRIPLE:
                                nitrous = False
                        if nitrous:
                            kegg_atom = "N0"
                        else:
                            kegg_atom = "N3a"
                    else:
                        kegg_atom = "N0"

            else:
                kegg_atom = "N0"

        kegg_atom_label = self._create_kegg_atom_label(kegg_atom)

        return kegg_atom_label

    def _get_label_O(self, atom):
        num_neighbor_nitrogen_atoms = 0
        num_neighbor_phosphorus_atoms = 0
        num_neighbor_sulfur_atoms = 0
        is_aldehyde = False
        num_neighbor_carbonyl_carbons = 0
        num_neighbor_carbons_neighbor_oxygens = 0
        is_sulfuric = False
        neighbor_carbon_is_in_ring = False
        has_double = False
        is_co2 = False
        is_formaldehyde = False
        is_nitro = False
        is_phosphric = False

        for bond in atom.GetBonds():
            if bond.GetEndAtom().GetSymbol() != "O":
                bond_partner = bond.GetEndAtom()
            else:
                bond_partner = bond.GetBeginAtom()
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                has_double = True
            if bond_partner.GetSymbol() == "C":
                neighbor_carbon_is_in_ring = self._check_in_ring(bond_partner)
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    if len(bond_partner.GetBonds()) == 2:
                        if bond_partner.GetExplicitValence() - \
                                bond_partner.GetNumExplicitHs() == 4:
                            is_co2 = True
                        is_aldehyde = True
                for bond2 in bond_partner.GetBonds():
                    if bond2.GetEndAtom().GetSymbol() != "C":
                        bond2_partner = bond2.GetEndAtom()
                    else:
                        bond2_partner = bond2.GetBeginAtom()

                    if bond2_partner.GetSymbol() == "O":
                        num_neighbor_carbons_neighbor_oxygens += 1
                        if bond2.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            num_neighbor_carbonyl_carbons += 1
                if len(bond_partner.GetBonds()) == 1:
                    is_formaldehyde = True
            elif bond_partner.GetSymbol() == "N":
                num_neighbor_nitrogen_atoms += 1
                if atom.GetFormalCharge() == -1:
                    is_nitro = True
            elif bond_partner.GetSymbol() == "P":
                num_p_r = 0
                num_neighbor_phosphorus_atoms += 1
                for neighbor2 in bond_partner.GetNeighbors():
                    if neighbor2.GetSymbol() == "O":
                        if neighbor2.GetExplicitValence() - \
                                neighbor2.GetNumExplicitHs() == 1:
                            is_phosphric = True
                    else:
                        num_p_r += 1
                if num_p_r + 4 - len(bond_partner.GetNeighbors()) >= 2:
                    is_phosphric = False
            elif bond_partner.GetSymbol() == "S":
                num_neighbor_sulfur_atoms += 1
                for neighbor2 in bond_partner.GetNeighbors():
                    if neighbor2.GetSymbol() == "O":
                        if neighbor2.GetExplicitValence() - \
                                neighbor2.GetNumExplicitHs() == 1:
                            is_sulfuric = True

        Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
        if num_neighbor_carbonyl_carbons == 0:
            if Explicit == 1:
                if num_neighbor_nitrogen_atoms == 1:
                    if is_nitro:
                        kegg_atom = "O3a"
                    else:
                        kegg_atom = "O1b"
                elif num_neighbor_phosphorus_atoms == 1:
                    kegg_atom = "O1c"
                elif num_neighbor_sulfur_atoms == 1:
                    kegg_atom = "O1d"
                else:
                    kegg_atom = "O1a"
            elif Explicit == 2:
                if has_double:
                    if num_neighbor_nitrogen_atoms == 1:
                        kegg_atom = "O3a"
                    elif num_neighbor_phosphorus_atoms == 1:
                        if is_phosphric:
                            kegg_atom = "O1c"
                        else:
                            kegg_atom = "O3b"
                    elif num_neighbor_sulfur_atoms == 1:
                        if is_sulfuric:
                            kegg_atom = "O1d"
                        else:
                            kegg_atom = "O3c"
                    else:
                        kegg_atom = "O0"
                else:
                    if self._check_in_ring(atom):
                        kegg_atom = "O2x"
                    else:
                        if num_neighbor_phosphorus_atoms == 0:
                            kegg_atom = "O2a"
                        elif num_neighbor_phosphorus_atoms == 1:
                            kegg_atom = "O2b"
                        elif num_neighbor_phosphorus_atoms == 2:
                            kegg_atom = "O2c"
                        else:
                            kegg_atom = "O0"
            else:
                kegg_atom = "O0"

        elif num_neighbor_carbonyl_carbons in [1, 2]:
            if is_co2:
                kegg_atom = "O0"
            else:
                if num_neighbor_carbons_neighbor_oxygens == 1:
                    if neighbor_carbon_is_in_ring:
                        kegg_atom = "O5x"
                    elif is_formaldehyde:
                        kegg_atom = "O0"
                    elif is_aldehyde:
                        kegg_atom = "O4a"
                    else:
                        kegg_atom = "O5a"
                elif num_neighbor_carbons_neighbor_oxygens in [2, 3, 4]:
                    if has_double:
                        kegg_atom = "O6a"
                    elif Explicit == 1:
                        kegg_atom = "O6a"
                    elif Explicit == 2:
                        if self._check_in_ring(atom):
                            kegg_atom = "O7x"
                        else:
                            kegg_atom = "O7a"
                    else:
                        kegg_atom = "O0"
                else:
                    kegg_atom = "O0"

        else:
            kegg_atom = "O0"

        kegg_atom_label = self._create_kegg_atom_label(kegg_atom)

        return kegg_atom_label

    def _get_label_S(self, atom):
        num_neighbor_oxygen_atoms = 0
        num_neighbor_sulfur_atoms = 0
        Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == "O":
                num_neighbor_oxygen_atoms += 1
            elif neighbor.GetSymbol() == "S":
                num_neighbor_sulfur_atoms += 1
        if self._check_in_ring(atom):
            if num_neighbor_sulfur_atoms > 0:
                kegg_atom = "S3x"
            else:
                kegg_atom = "S2x"
        else:
            if Explicit == 1:
                kegg_atom = "S1a"
            elif num_neighbor_oxygen_atoms > 0:
                kegg_atom = "S4a"
            elif num_neighbor_sulfur_atoms > 0:
                kegg_atom = "S3a"
            elif len(atom.GetNeighbors()) == 2:
                kegg_atom = "S2a"
            else:
                kegg_atom = "S0"

        kegg_atom_label = self._create_kegg_atom_label(kegg_atom)

        return kegg_atom_label

    def _get_label_P(self, atom):
        num_neighbor_oxygen_atoms = 0
        num_neighbor_sulfur_atoms = 0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == "O":
                num_neighbor_oxygen_atoms += 1
            elif neighbor.GetSymbol() == "S":
                num_neighbor_sulfur_atoms += 1
        if num_neighbor_sulfur_atoms == 0 and num_neighbor_oxygen_atoms >= 3:
            kegg_atom = "P1b"
        else:
            kegg_atom = "P1a"

        kegg_atom_label = self._create_kegg_atom_label(kegg_atom)

        return kegg_atom_label

    def _get_label_else(self, atom):
        if atom.GetSymbol() in ["F", "Cl", "Br", "I"]:
            kegg_atom_label = self._create_kegg_atom_label("X", atom)
        elif atom.GetSymbol() == "*":
            kegg_atom_label = self._create_kegg_atom_label("R")
        else:
            kegg_atom_label = self._create_kegg_atom_label("Z", atom)

        return kegg_atom_label

    def _check_in_ring(self, atom):
        """
        RDKit の atom object を受け取り，その atom が ring に含まれているかを判別する。
        atom.IsInRing という method も存在するが，kcf の仕様上，21 員環以上のものは
        ring とみなさないため，atom.IsInRingSize で検証する。
        """
        for ring_size in range(3, 21):
            if atom.IsInRingSize(ring_size):
                return True

        return False

    def _create_kegg_atom_label(self, kegg_atom, atom=None):
        """
        kegg_atom を受け取って，kcf に変換しやすい dict として返す。
        kegg_atom 自体は文字列。
        また，kegg_atom の文字列は 3文字以内である。
        """
        label = dict()
        if atom is None:
            label["atom_species"] = kegg_atom[0]
        else:
            if atom.GetSymbol()[0] != "R":
                label["atom_species"] = atom.GetSymbol()
            else:
                label["atom_species"] = "R#"
        label["atom_class"] = kegg_atom[0:2]
        label["kegg_atom"] = kegg_atom

        return label

    def convert_kcf_vec(self, levels=list(range(3)), attributes=list(range(6)),
                        max_sub_length=12, max_ring_size=9):
        """
        rdkit の mol file から生成した networkx の graph のうち，
        炭素同士以外の edge を切り，それぞれの subgraphs を取り，
        その内で， node 数が 4 以上のものをそれぞれ，skeleton, inorganic として記録する。
        self._get_pin_path でそれぞれの pin_path を生成。
        self.kegg_atom_label を用いて
        """
        s_skeleton = set()
        s_inorganic = set()
        c_graph = deepcopy(self.graph)
        c_graph_edges = deepcopy(self.graph.edges())
        for edge in c_graph_edges:
            ele1 = self.get_symbol(edge[0])
            ele2 = self.get_symbol(edge[1])
            if (ele1 == "C" and ele2 != "C") or (ele1 != "C" and ele2 == "C"):
                c_graph.remove_edge(edge[0], edge[1])
        subgraphs = [c_graph.subgraph(c).copy() for c in nx.connected_components(c_graph)]
        for subgraph in subgraphs:
            if len(subgraph.nodes()) < 4:
                continue
            s_subgraph = ",".join(list(map(str, sorted(subgraph.nodes()))))
            if subgraph.nodes()[min(list(subgraph.nodes()))]["symbol"] == "C":
                s_skeleton.add(s_subgraph)
            else:
                s_inorganic.add(s_subgraph)

        pin_path_1 = []
        for subgraph in subgraphs:
            pin_path = self._get_pin_path(subgraph)
            pin_path_1.append(pin_path)
        pin_path_2 = []
        pin_path_3 = []
        for cutoff in range(4, max_sub_length + 1):
            pin_path = self._get_pin_path(self.graph, cutoff)
            pin_path_3.append([cutoff, pin_path])
            pin_path_2_sub = []
            for subgraph in subgraphs:
                pin_path = self._get_pin_path(subgraph, cutoff)
                pin_path_2_sub.append([cutoff, pin_path])
            pin_path_2.append(pin_path_2_sub)

        kegg_atom_keys = ["atom_species", "atom_class", "kegg_atom"]
        for level in levels:
            kegg_atom_key = kegg_atom_keys[level]
            if 0 in attributes:
                for atom in self.kegg_atom_label.values():
                    self._add_vec_element(atom[kegg_atom_key], 1, "atom",
                                          kegg_atom_key)
            if 1 in attributes:
                for bond in self.graph.edges():
                    l_atoms = [self.kegg_atom_label[i][kegg_atom_key]
                               for i in bond]
                    ele = "-".join(sorted(l_atoms))
                    self._add_vec_element(ele, 2, "bond", kegg_atom_key)

            if 2 in attributes:
                for triplet in self.get_triplets():
                    l_ele = [self.kegg_atom_label[i][kegg_atom_key]
                             for i in triplet]
                    ele_1 = "-".join(l_ele)
                    ele_2 = "-".join(reversed(l_ele))
                    ele = sorted([ele_1, ele_2])[0]
                    self._add_vec_element(ele, 3, "triplet", kegg_atom_key)

            if 3 in attributes:
                for vicinity in self.get_vicinities():
                    if len(vicinity[1]) < 3:
                        continue
                    vic_tmp = [self.kegg_atom_label[i][kegg_atom_key]
                               for i in vicinity[1]]
                    vic_tmp = sorted(vic_tmp)
                    l_ele = [vic_tmp[0],
                             self.kegg_atom_label[vicinity[0]][kegg_atom_key],
                             vic_tmp[1]]
                    ele = "-".join(l_ele)
                    for i in range(2, len(vic_tmp)):
                        ele += ",2-" + vic_tmp[i]
                    self._add_vec_element(ele, 1 + len(vicinity[1]),
                                          "vicinity", kegg_atom_key)

            self.ring_string.append(dict())
            if 4 in attributes:
                for ring in self.find_seq(max_ring_size + 1):
                    if len(ring) < 4:
                        continue
                    if ring[0] != ring[-1]:
                        continue
                    ring_str = ",".join(list(map(str, sorted(ring[0:-1]))))
                    l_ele = [self.kegg_atom_label[i][kegg_atom_key]
                             for i in ring[0:-1]]
                    ele = "-".join(l_ele)
                    for i in range(len(ring) - 2):
                        for j in range(i + 2, len(ring) - 1):
                            if self.has_bond(ring[i], ring[j]):
                                ele += "," + str(i + 1) + "-" + str(j + 1)
                    if ring_str not in self.ring_string[-1].keys():
                        self.ring_string[-1][ring_str] = ele
                    elif self.ring_string[-1][ring_str] > ele:
                        self.ring_string[-1][ring_str] = ele
                for ring_str, ele in self.ring_string[-1].items():
                    self._add_vec_element(ele, len(ring_str.split(",")),
                                          "ring", kegg_atom_key)

            self.subs_string.append(dict())
            if 5 in attributes:
                for pin_path_1_x in pin_path_1:
                    for ring_str, ele in \
                            self._bund_pin_path(pin_path_1_x,
                                                kegg_atom_key).items():
                        if ring_str not in self.subs_string[-1].keys():
                            self.subs_string[-1][ring_str] = ele
                        elif self.subs_string[-1][ring_str] > ele:
                            self.subs_string[-1][ring_str] = ele
                for pin_path_2_x in pin_path_2:
                    for cut_off, pin_path_2_y in pin_path_2_x:
                        for ring_str, ele in \
                                self._bund_pin_path(pin_path_2_y,
                                                    kegg_atom_key,
                                                    cutoff).items():
                            if ring_str not in self.subs_string[-1].keys():
                                self.subs_string[-1][ring_str] = ele
                            elif self.subs_string[-1][ring_str] > ele:
                                self.subs_string[-1][ring_str] = ele
                for cut_off, pin_path_3_x in pin_path_3:
                    for ring_str, ele in \
                            self._bund_pin_path(pin_path_3_x,
                                                kegg_atom_key,
                                                cut_off).items():
                        if ring_str not in self.subs_string[-1].keys():
                            self.subs_string[-1][ring_str] = ele
                        elif self.subs_string[-1][ring_str] > ele:
                            self.subs_string[-1][ring_str] = ele

                for k, s in self.subs_string[-1].items():
                    if len(k.split(",")) < 4:
                        continue
                    if k in self.ring_string[-1].keys():
                        continue
                    elif k in s_skeleton:
                        self._add_vec_element(s, len(k.split(",")),
                                              "skeleton", kegg_atom_key)
                    elif k in s_inorganic:
                        self._add_vec_element(s, len(k.split(",")),
                                              "inorganic", kegg_atom_key)
                    elif len(s.split(",")) == 1:
                        self._add_vec_element(s, len(k.split(",")),
                                              "linear", kegg_atom_key)
                    else:
                        self._add_vec_element(s, len(k.split(",")),
                                              "unit", kegg_atom_key)

        return True

    def _get_pin_path(self, graph, cut_off=None):
        """
        TODO
        """
        head_pin = defaultdict(list)
        for node_1 in graph.nodes():
            for node_2 in graph.nodes():
                l_sequences = []
                if cut_off:
                    for sequence in nx.all_simple_paths(graph, node_1, node_2,
                                                        cutoff=cut_off):
                        if len(sequence) > cut_off:
                            if sequence[0] != sequence[-1]:
                                continue
                        l_sequences.append(sequence)
                else:
                    l_sequences = list(nx.all_simple_paths(graph, node_1,
                                                           node_2))
                for sequence in l_sequences:
                    if (len(sequence) == 3) and (sequence[0] == sequence[-1]):
                        continue
                    head = "-".join(list(map(str, sequence[:2])))
                    if node_1 == node_2:
                        head_pin[head].append([1 - len(sequence), sequence])
                    else:
                        head_pin[head].append([-len(sequence), sequence])

        return head_pin

    def _add_vec_element(self, ele, n_nodes, ele_type, ele_level):
        """
        kcf_vec の要素登録。
        """
        if ele not in self.kcf_vec.keys():
            self.kcf_vec[ele]["n_nodes"] = n_nodes
            self.kcf_vec[ele]["ele_type"] = ele_type
            self.kcf_vec[ele]["ele_level"] = ele_level
            self.kcf_vec[ele]["count"] = 1
        else:
            self.kcf_vec[ele]["count"] += 1

        return True

    def _bund_pin_path(self, head_pin, level, cutoff=None):
        """
        TODO
        """
        d_skeleton = dict()
        for length_seq in head_pin.values():
            l_seen = []
            string = ""
            idx = 0
            prev_idx = 0
            prev_not_seen = False
            main_chain = set()
            bridges = set()
            length_kegg_seq = []
            for length, seq in length_seq:
                l_labels = [self.kegg_atom_label[atom][level] for atom in seq]
                length_kegg_seq.append([length, l_labels, seq])

            for length, l_labels, seq in sorted(length_kegg_seq):
                if seq[0] != seq[-1]:
                    key = ",".join(list(map(str, sorted(seq))))
                    value = "-".join(l_labels)
                    if key not in d_skeleton.keys():
                        d_skeleton[key] = value
                    elif d_skeleton[key] > value:
                        d_skeleton[key] = value
                idx += 1
                for i, atom in enumerate(seq):
                    if atom in l_seen:
                        if i != 0:
                            if prev_not_seen:
                                string += "-" + str(l_seen.index(atom) + 1)
                                chain = sorted([l_seen.index(seq[i - 1]) + 1,
                                                l_seen.index(atom) + 1])
                                main_chain.add(tuple(chain))
                                break
                            else:
                                bridge = sorted([l_seen.index(seq[i - 1]) + 1,
                                                 l_seen.index(atom) + 1])
                                if tuple(bridge) not in main_chain:
                                    if tuple(bridge) not in bridges:
                                        bridges.add(tuple(bridge))
                        prev_not_seen = False
                    else:
                        if i != 0 and idx != prev_idx:
                            string += "," + str(l_seen.index(seq[i - 1]) + 1)
                        if len(string) > 0:
                            string += "-"
                        string += self.kegg_atom_label[atom][level]
                        l_seen.append(atom)
                        if i != 0:
                            chain = sorted([l_seen.index(seq[i - 1]) + 1,
                                            l_seen.index(atom) + 1])
                            main_chain.add(tuple(chain))
                        prev_idx = idx
                        prev_not_seen = True
            if len(l_seen) <= 4:
                continue
            if len(bridges) > 0:
                l_bridge = []
                for bridge in sorted(list(bridges)):
                    s_bridge = "-".join(list(map(str, bridge)))
                    l_bridge.append(s_bridge)
                string += "," + ",".join(l_bridge)
            seen = ",".join(list(map(str, sorted(l_seen))))
            if seen not in d_skeleton.keys():
                d_skeleton[seen] = string
            elif d_skeleton[seen] > string:
                d_skeleton[seen] = string

        return d_skeleton

    def get_pandas_df(self):
        matrix = []
        for key, value in self.kcf_vec.items():
            ele = [key, value["ele_type"], value["ele_level"],
                   value["count"]]
            matrix.append(ele)
        columns = ["str", "type", "level", "count"]
        df = pd.DataFrame(sorted(matrix, key=lambda x: x[3], reverse=True),
                          columns=columns)
        return df

    def string2seq(self, kcfstring):
        if len(self._string_2_seq) == 0:
            dict1 = dict()
            for string_dicts in [self.subs_string, self.ring_string]:
                for dict2 in self.subs_string:
                    #dict1 = dict()
                    for k, v in dict2.items():
                        if v not in dict1.keys():
                            dict1[v] = []
                        if k not in dict1[v]:
                            dict1[v].append(k)
                    #ret_list.append(dict1)
            self._string_2_seq = dict1
        if kcfstring in self._string_2_seq.keys():
            return self._string_2_seq[kcfstring]
        else:
            return []
    
    def draw_cpd_with_highlighted_substructure(self, subs_string="C-C-C-C", image_file="mol.svg", height=300, width=500, 
                                               highlightAtoms=[], highlightAtomColors={}, highlightAtomRadii={}, start=0, custom_label=None):
        highlightAtoms = list(set([int(atom_idx) for atom_idx in ",".join(self.string2seq(subs_string)).split(",")]))
        return self.draw_cpd_with_labels(image_file = image_file, height=height, width=width, highlightAtoms=highlightAtoms, \
                                  highlightAtomColors=highlightAtomColors, highlightAtomRadii=highlightAtomRadii, start=start, custom_label=custom_label)



