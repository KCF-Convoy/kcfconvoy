# coding: utf-8

import networkx as nx
from .Compound import Compound
from rdkit import Chem
from rdkit.Chem import AllChem


class Library:
    """
    method
        - calc_fingerprints
            fingerprint を計算する。
            fingerprint の種類は，
                - PatternFingerprint
                - RDKFingerprint
                - MorganFingerprint
                - LayeredFingerprint
                - PatternFingerprint
            fingerprintで種類を渡す。default値はPatternFingerprint
        - input_from_kegg
            KEGG_IDからobjectを生成する。
            - input:
                kegg_id
                name
        - input_from_knapsack
            knapsackからobjectを生成する。
            - input:
                knapsack_id
                name
        - input_molfile
            molfileのpathからobjectを生成する。
            - input:
                molfile_path
                name
        - input_inchi
            inchi形式からobjectを生成する。
            - input:
                inchi
                name
        - input_smiles
            smiles形式からobjectを生成する。
            - input:
                smiles
                name
        - input_rdkmol
            rdkitのmol形式からobjectを生成する。
            - input:
                rdk_mol
                name

    内部 method
        - _append_cpd
            input系の内部メソッド，Libraryにcpdを加える。
    """

    def __init__(self):
        """
        - self.cpds
            Compound クラスのリスト。
        """
        self.cpds = []
        self.digraph = nx.DiGraph()
        self.generations = []
        self.inchis = []
        self.names = []
        self.fps = []

    def calc_fingerprints(self, fingerprint="PatternFingerprint"):
        """
        fingerprint を計算する。

        fingerprint の種類は，
            - PatternFingerprint
            - RDKFingerprint
            - MorganFingerprint
            - LayeredFingerprint
            - PatternFingerprint

        http://cheminformist.itmol.com/TEST/wp-content/uploads/2015/08/FpScreening_a.html
        """
        for cpd in self.cpds:
            if fingerprint == "PatternFingerprint":
                fp = Chem.PatternFingerprint(cpd.mol, fpSize=1024)
            elif fingerprint == "RDKFingerprint":
                fp = Chem.RDKFingerprint(cpd.mol)
            elif fingerprint == "MorganFingerprint":
                fp = AllChem.GetMorganFingerprintAsBitVect(cpd.mol, 2)
            elif fingerprint == "LayeredFingerprint":
                fp = Chem.LayeredFingerprint(
                    cpd.mol,
                    layerFlags=Chem.LayeredFingerprint_substructLayers)
            elif fingerprint == "PatternFingerprint":
                fp = Chem.PatternFingerprint(cpd.mol, fpSize=1024)

            self.fps.append(fp)

        return True

    def _append_cpd(self, cpd, name):
        """
        input系の内部メソッド，Libraryにcpdを加える。
        """
        if name is None:
            name = len(self.cpds)
        self.digraph.add_node(len(self.cpds))

        try:
            inchis = Chem.MolToInchi(cpd.mol)
        except:
            return False    # TODO error 処理

        self.inchis.append(inchis)
        self.cpds.append(cpd)
        self.generations.append(0)
        self.names.append(name)

        return True

    def input_from_kegg(self, cid, name=None):
        """
        e.g., cid = C00002
        KEGG_IDからmol形式でdownloadし，./kegg/cid.molとして保存する。
        """
        cpd = Compound()
        cpd.input_from_kegg(cid)
        self._append_cpd(cpd, name)

        return True

    def input_from_knapsack(self, cid, name=None):
        """
        e.g., cid = C00002657
        knapsack_IDからmol形式でdownloadし，./knapsack/cid.molとして保存する。
        """
        cpd = Compound()
        cpd.input_from_knapsack(cid)
        self._append_cpd(cpd, name)

        return True

    def input_molfile(self, molfile, name=None):
        """
        molfileの実体ではなく，molfileのpathを受け取る。
        """
        cpd = Compound()
        cpd.input_molfile(molfile)
        self._append_cpd(cpd, name)

        return True

    def input_inchi(self, inchi, name=None):
        """
        inchi形式を受け取って，rdkitのmol形式に直し，mol形式をmolblockに直し登録する。
        """
        cpd = Compound()
        cpd.input_inchi(inchi)
        self._append_cpd(cpd, name)

        return True

    def input_smiles(self, smiles, name=None):
        """
        smiles形式を受け取って，rdkitのmol形式に直し，mol形式をmolblockに直し登録する。
        """
        cpd = Compound()
        cpd.input_smiles(smiles)
        self._append_cpd(cpd, name)

        return True

    def input_rdkmol(self, mol, name=None):
        """
        rdkitのmol形式を受け取り，mol形式をmolblockに直し登録する。
        """
        cpd = Compound()
        cpd.input_rdkmol(mol)
        self._append_cpd(cpd, name)

        return True
