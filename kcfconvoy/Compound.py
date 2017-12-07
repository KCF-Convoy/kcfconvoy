from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx
import os
import urllib.request
class Compound: # KCF-Convoy
    """ Daylight Molfile and SDF http://infochim.u-strasbg.fr/recherche/Download/Fragmentor/MDL_SDF.pdf (2010) """
    """ or http://www.daylight.com/meetings/mug05/Kappler/ctfile.pdf (2003) """
    """ RDKit http://www.rdkit.org/GettingStartedInPython.pdf (2012) """
    """ NetworkX 2.0 https://networkx.github.io/documentation/development/_downloads/networkx_reference.pdf """
    """ NetworkX 1.9.1 https://networkx.github.io/documentation/networkx-1.9.1/_downloads/networkx_reference.pdf """
    """ NetworkX 1.9 https://networkx.github.io/documentation/networkx-1.9/_downloads/networkx_reference.pdf """ 
    def __init__(self):
        self.heads = []
        self.tails = []
        self.fit2d = False
        self.graph = nx.Graph()
        self.mol = None
    
    def draw_cpd(self, imagefile="mol.png", shownum=False):
        if not self.fit2d:
            rdDepictor.Compute2DCoords(self.mol)
            self.set_coordinates()
            self.fit2d = True
        if shownum:
            mol = Chem.MolFromMolBlock(self.get_molblock(shownum=True))
            Draw.MolToFile(mol, imagefile)
        else:
            Draw.MolToFile(self.mol, imagefile)
        print(imagefile)
        return True

    def draw_cpd_with_custom_labels(self, custom_label):
        if not self.fit2d:
            rdDepictor.Compute2DCoords(self.mol)
            self.set_coordinates()
            self.fit2d = True
        pos = self.get_coordinates()
        node_color = self.get_node_colors()
        node_label = custom_label
        edge_labels = {}
        for edge in self.graph.edges(data=True):
            edge_labels[(edge[0], edge[1])] = edge[2]['index']
        nx.draw(self.graph, pos, node_color=node_color, alpha=0.4)
        nx.draw_networkx_labels(self.graph,pos,fontsize=6, labels=node_label)
        #nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, fontsize=3, font_color='b')
        plt.draw()

    def find_seq(self, length, bidirectonal=True):
        for i in self.graph.nodes():
            for j in self.graph.nodes():
                if (length == 3) and (i == j):
                    continue
                if bidirectonal:
                    pass
                elif i > j:
                    continue
                for seq in nx.all_simple_paths(self.graph, i, j, cutoff=(length + 1)):
                    if len(seq) == 2:
                        pass
                    elif len(seq) <= length:
                        yield seq
                    elif len(seq) == length + 1:
                        if seq[0] == seq[-1]:
                            yield seq
                        continue

    def get_coordinates(self):
        return [[float(x) for x in node[1]['row'][0:2]] for node in self.graph.nodes(data=True)]

    def get_node_colors(self):
        return self.label_to_nums([node[1]['symbol'] for node in self.graph.nodes(data=True)])

    def input_from_kegg(self, cid): # e.g., cid = C00002 
        kegg_dir = "kegg"
        if not os.path.exists(kegg_dir):
            os.mkdir(kegg_dir)
        if not os.path.exists("%s/%s.mol" % (kegg_dir, cid)):
            url = "http://www.genome.jp/dbget-bin/www_bget?-f+m+%s" % (cid)
            urllib.request.urlretrieve(url, "%s/%s.mol" % (kegg_dir, cid))
        self.input_molfile("%s/%s.mol" % (kegg_dir, cid))
        return True

    def input_inchi(self, inchi):
        self.input_rdkmol(Chem.MolFromInchi(inchi))
        return True

    def input_molblock(self, molblock):
        n_atoms = 0; n_bonds = 0
        replace = []
        for i, line in enumerate(molblock.split("\n")):
            if i < 4:
                if i == 3:
                    n_atoms = int(line[:3])
                    n_bonds = int(line[3:6])
                self.heads.append(line)
            elif i < (4 + n_atoms):
                a = line.split()
                if a[12] == '0':
                    a[12] = str(len(self.graph.nodes()) + 1) 
                if self.fit2d or float(a[0]) != 0.0:
                    self.fit2d = True
                if a[3] == 'H': #####
                    continue #####
                replace.append(i - 4) #atomid[str(i - 3)] = len(self.graph.nodes())
                self.graph.add_node(len(self.graph.nodes()), symbol = a[3], row=a)
            elif i < (4 + n_atoms + n_bonds):
                a = [line[0:3].strip(), line[3:6].strip(), line[6:9].strip(), line[9:12].strip()]
                if a[2] == '2': a[3] = '0'
                bondidx = i - 4 - n_atoms
                if int(a[0]) - 1 not in replace:
                    continue
                if int(a[1]) - 1 not in replace:
                    continue
                b = [s for s in a]
                b[0] = str(replace.index(int(a[0]) - 1) + 1)
                b[1] = str(replace.index(int(a[1]) - 1) + 1)
                self.graph.add_edge(replace.index(int(a[0]) - 1), # replaceatomid[a[0]], replaceatomid[a[1]], 
                                    replace.index(int(a[1]) - 1), order=int(a[2]), index=bondidx, row=b)
            else:
                if 'CHG' in line.split(): continue
                self.tails.append(line)
        self.n_atoms = n_atoms
        self.n_bonds = n_bonds
        return True

    def input_molfile(self, molfile):
        with open(molfile) as f:
            molblock = f.read()
            self.input_molblock(molblock)
            self.mol = Chem.MolFromMolBlock(molblock)
        return True

    def input_rdkmol(self, mol):
        self.mol = mol
        self.input_molblock(Chem.MolToMolBlock(mol))

    def has_bond(self, atm1, atm2):
        return atm1 in self.graph.adj[atm2].keys()

    def label_to_nums(self, lst):
        i = 0
        seen = []
        dic = {}
        for label in sorted(lst):
            if label not in seen:
                seen.append(label)
                dic[label] = i
                i += 1
        return [dic[label] for label in lst]

    def symbol(self, atmidx):
        return self.graph.node[atmidx]['symbol']

    def triplets(self):
        return self.find_seq(3, bidirectonal=False)

    def vicinities(self):
        return [(j, [i for i in nx.all_neighbors(self.graph, j)]) 
                for j in self.graph.nodes()]



