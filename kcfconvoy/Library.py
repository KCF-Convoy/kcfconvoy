# KCF-Convoy
from IPython.utils.io import rprint
from rdkit import Chem
from rdkit.Chem import AllChem
from kcfconvoy.Compound import Compound
from time import sleep
import networkx as nx
import os
import urllib.request
class Library:
    """This class uses the Compound class"""
    def __init__(self):
        self.cpds = []
        self.digraph = nx.DiGraph()
        self.generations = []
        self.inchis = []
        self.names = []

    def calc_fingerprints(self, x=0):
        if x == 0:
            self.fps = [Chem.PatternFingerprint(cpd.mol, fpSize=1024) for cpd in self.cpds]
        elif x == 1:
            self.fps = [Chem.RDKFingerprint(cpd.mol) for cpd in self.cpds]
        elif x == 2:
            self.fps = [AllChem.GetMorganFingerprintAsBitVect(cpd.mol,2) for cpd in self.cpds]
        elif x == 3:
            self.fps = [Chem.LayeredFingerprint(cpd.mol,layerFlags=Chem.LayeredFingerprint_substructLayers) for cpd in self.cpds]
        elif x == 4:
            self.fps = [Chem.LayeredFingerprint(cpd.mol,layerFlags=Chem.LayeredFingerprint_substructLayers) for cpd in self.cpds]
        else:
            self.fps = [Chem.PatternFingerprint(cpd.mol, fpSize=1024) for cpd in self.cpds]
        return True

    def input_from_kegg(self, cid, wait=1): # e.g., cid = C00002
        kegg_dir = "kegg"
        if not os.path.exists(kegg_dir):
            os.mkdir(kegg_dir)
        if not os.path.exists("%s/%s.mol" % (kegg_dir, cid)):
            url = "http://www.genome.jp/dbget-bin/www_bget?-f+m+compound+%s" % (cid)
            urllib.request.urlretrieve(url, "%s/%s.mol" % (kegg_dir, cid))
            sleep(wait)
            rprint("%s/%s.mol downloaded."  % (kegg_dir, cid))
        return self.input_molfile("%s/%s.mol" % (kegg_dir, cid), cid)

    def input_inchi(self, inchi, name=False):
        self.input_rdkmol(Chem.MolFromInchi(inchi), name)
        return True

    def input_molfile(self, molfile, name=False):
        if not name:
            name = len(self.cpds)
        self.digraph.add_node(len(self.cpds))
        c = Compound()
        lines = ''
        with open(molfile) as f:
            lines += f.read()
        c.input_molblock(lines)
        c.mol = Chem.MolFromMolBlock(lines)
        #self.inchis.append(Chem.MolToInchi(c.mol))
        try:
            self.inchis.append(Chem.MolToInchi(c.mol))
        except:
            return False
        self.cpds.append(c)
        self.generations.append(0)
        self.names.append(name)
        return True

    def input_rdkmol(self, mol, name=False):
        if not name:
            name = len(self.cpds)
        self.digraph.add_node(len(self.cpds))
        c = Compound()
        c.mol = mol
        c.input_molblock(Chem.MolToMolBlock(mol))
        self.cpds.append(c)
        self.inchis.append(Chem.MolToInchi(mol))
        self.generations.append(0)
        self.names.append(name)
        return True

    def input_smiles(self, smi, name=False):
        self.input_rdkmol(Chem.MolFromSmiles(smi), name)
        return True
