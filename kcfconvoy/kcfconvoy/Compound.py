# coding: utf-8

import os
import urllib.request
from copy import copy

import matplotlib.pyplot as plt

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

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
            generates a compound using KEGG compound ID. See https://www.kegg.jp
            - input:
                kegg_id
        - input_from_knapsack
            generates a compound using KNApSAcK compound ID. See http://kanaya.naist.jp/KNApSAcK/
            - input:
                knapsack_id
        - input_molfile
            generates a compound from a molfile specified by molfile_path
            - input:
                molfile_path
        - input_inchi
            generates a compound from a InChI string.
            - input:
                inchi
        - input_smiles
            generates a compound from a SMILES string.
            - input:
                smiles
        - input_rdkmol
            generates a compound from an RDK mol object.
            - input:
                rdk_mol
        - draw_cpd
            depicts the compound as png_file_name.png in the working directory using RDKit.
            - input:
                png_file_name
        - draw_cpd_with_labels
            depicts the compound with atom labelings. The users can also specify the start labels or customize the labels.
        - find_seq
            an iterator that yields the substructures (the sequence of atoms) with the length equals or shorter than specified.
            To parameters: length, and bidirectonal (optional).
            - output:
                list of the atom IDs.
        - has_bond
            returns a bool value representing whether or not the compound has a chemical bond between atom_1 and atom_2.
            - input:
                atom_1
                atom_2
        - get_symbol
            returns the symbol of the atomic element specified by the atom ID.
            - input:
                atom_index
        - get_triplets
            returns the substructures that consists of three atoms
        - get_vicinities
            returns the substructures that consists of a center atom and all the attaching atoms.

    internal (hidden) method:
        - _input_molblock
            is called by input_* methods and generates self.graph.
        - _compute_2d_coords
            is called when the compound does not have any 2D coordinates and generates 2D coordinates and call self._set_coordinates().
        - self._set_coordinates
            sets the 2D coordinates to self.graph.nodes
        - self._get_coordinates
            returns the 2D coordinates
        - _get_node_colors
            is called by draw_cpd_with_labels to determine the node colors.
        - _get_node_labels
            is called by draw_cpd_with_labels to label the nodes.
    """

    def __init__(self):
        """
        self.heads : The third row in the Molfile format (0-indexed)
            ex.)   4  4  0  0  0  0  0  0  0  0999 V2000
        self.n_atoms : The number of atoms defined in the atom block in Molfile, 
            which is contained in self.graph
        self.n_bonds : The number of bonds defined in the bond block in Molfile, 
            which is contained in self.graph
        self.tail : The rows after the bond block in the Molfile format
        self.fit2d : True if 2D corrdinates are already calculated, otherwize False.
            2D coordinates are described in self.graph[node]["row"]
        self.graph : molecular graph structure by NetworkX
            a node represents an atom and has metadata ["symbol", "row"]
            an edge represents a bond and has metadata ["order", "index", "row"]
        self.mol : RDK mol calculated by RDKit
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
        e.g., cid = "C00002"
        This method downloads a Molfile specified by cid (KEGG Compound ID) from KEGG database,
        save it as ./kegg/cid.mol, and calls self.input_molfile to generate a Compound object.
        If ./kegg/cid.mol already exists, this method does not download the same file again.
        """
        kegg_dir = "kegg"
        if not os.path.isdir("./" + kegg_dir):
            os.mkdir("./" + kegg_dir)
        if not os.path.isfile("./{}/{}.mol".format(kegg_dir, cid)):
            url = "http://www.genome.jp/dbget-bin/www_bget?-f+m+{}".format(cid)
            print("Downloading ", cid)
            urllib.request.urlretrieve(url,
                                       "./{}/{}.mol".format(kegg_dir, cid))

        return self.input_molfile("./{}/{}.mol".format(kegg_dir, cid))

    def input_from_knapsack(self, cid):
        """
        e.g., cid = "C00002657"
        This method downloads a Molfile specified by cid (KNApSAcK ID) from KNApSAcK 3D database,
        save it as ./knapsack/cid.mol, and calls self.input_molfile to generate a Compound object.
        If ./knapsack/cid.mol already exists, this method does not download the same file again.
        The molfile from KNApSAcK contains 3D coordinates, so the 2D coordinates are recalculated. 
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
        This method inputs the path to the Molfile (not the Molfile itself). 
        The data in the Molfile is referred to as molblock.
        This method calls self._input_molblock to incorporate varias information into self.
        If 2D corrdinate is not calculated yet, this method calls self._compute_2d_coords().
        """
        with open(molfile, "r") as f:
            molblock = f.read()

        if not self._input_molblock(molblock):
            return False
        self.mol = Chem.MolFromMolBlock(molblock)

        if self.fit2d is False:
            self._compute_2d_coords()

        return True

    def input_inchi(self, inchi):
        """
        inputs an InChI string to generate an RDKit mol,
        which is passed to self.input_rdkmol() where a molblock is generated.
        """
        self.input_rdkmol(Chem.MolFromInchi(inchi))

        return True

    def input_smiles(self, smiles):
        """
        inputs a SMILES string to generate an RDKit mol,
        which is passed to self.input_rdkmol() where a molblock is generated.
        """
        self.input_rdkmol(Chem.MolFromSmiles(smiles))

        return True
    
    def input_rdkmol(self, rdkmol):
        """
        inputs an RDKit mol and generates a molblock.
        If 2D corrdinate is not calculated yet, this method calls self._compute_2d_coords().
        """
        self.mol = rdkmol
        self._input_molblock(Chem.MolToMolBlock(rdkmol))

        if self.fit2d is False:
            self._compute_2d_coords()

        return True

    def _input_molblock(self, molblock):
        """
        A molblock (the data in Molfile)  has the following format:
            The first three rows (0, 1, 2) are not used for the description of the chemical structure.
            The third row (0-indexed) contains n_atoms and n_edges, and is defined as self.head
            The (3+1)th - (3+1+n_atoms-1)th rows are referred to as the atom block, containing the data about atoms. 
                ex:) The 4-7 th rows are in the atom block if n_atoms=4
            The (3+1+n_atoms) - (3+1+n_atoms+n_edges-1)th rows are referred to as the bond block, containing the data about bonds.
                ex.) The 8 - 11 th rows are in the bond block if n_atoms=4,n_edges=4
            The (3+1+n_atoms+n_edges)th row and the following rows are appended to self.tail unless the row contains "CHG".

        The first column of the atom block represents X-coordinates of the atom, 
        meaning that 2D coordinates are already calculated if it is not 0.0

        Most information obtained from molblock is incorporated into self.head, self.tail, self.graph[node]["row"] etc.
        """
        l_molblock = molblock.split("\n")
        # The third column (0-indexed)
        try:
            line = l_molblock[3]
        except:
            return False
        self.n_atoms, self.n_bonds = \
            map(int, [line[:3].strip(), line[3:6].strip()])
        self.head = l_molblock[3]

        index_check_list = []
        # atom block
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

        # bond block
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
        calculates the 2D coordinates, and set them to self by calling self.set_coordinates()
        This method is called if self.fit2d == False in
        input_from_kegg, input_from_knapsack, input_inchi, input_rdkmol, and input_molfile.
        """
        AllChem.Compute2DCoords(self.mol)
        self._set_coordinates()
        self.fit2d = True

        return True

    def _set_coordinates(self):
        """
        obtains molblock from self.mol, extracts 2D coordinates from the molblock, and sets them to self.graph.node.
        Before using this method:
        - 2D coordinates must be calculated by AllChem.Compute2DCoords(self.mol) or _compute_2d_coords(self)
        - self._input_molblock() must be called to prepare self.graph[node]["row"] in advance.
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
        The "row" attribute of self.graph.nodes contain the row in the atom block in the molblock,
        where the 0th and 1st columns (0-indexed) contains X and Y coordinates. 
        This method extracts these 2D coorinates from self.graph.nodes as the nexted list.
        """
        l_rows = [node[1]["row"] for node in self.graph.nodes(data=True)]
        l_coordinates = [list(map(float, row[0:2])) for row in l_rows]

        return l_coordinates

    def draw_cpd(self, image_file="mol.svg", height=300, width=500, highlightAtoms=[], highlightAtomColors={},
                  highlightAtomRadii={}):
        """
        depicts the compound and saves it as image_file in the working directory.
        If the 2D coordinates are not calculated yet, this method calls self._compute_2d_coords().
        """
        if not self.fit2d:
            self._compute_2d_coords()

        #Draw.MolToFile(self.mol, image_file)
        view = rdMolDraw2D.MolDraw2DSVG(height,width)
        tm = rdMolDraw2D.PrepareMolForDrawing(self.mol)
        view.SetFontSize(0.9*view.FontSize())
        option = view.drawOptions()
        #option.atomLabels[6] = 'N1'
        #option.atomLabels[8] = 'N2'
        #option.multipleBondOffset=0.07
        #option.padding=0.11
        #option.legendFontSize=20
        view.DrawMolecule(tm, highlightAtoms=highlightAtoms, highlightAtomColors=highlightAtomColors,
                  highlightAtomRadii=highlightAtomRadii)
        view.FinishDrawing()
        svg = view.GetDrawingText()
        with open(image_file, 'w') as f:
            f.write(svg)
        return SVG(svg.replace('svg:', ''))

    def draw_cpd_with_labels(self, image_file="mol.svg", height=300, width=500, highlightAtoms=[], highlightAtomColors={},
                  highlightAtomRadii={}, start=0, custom_label=None):
        """
        depicts the compound with node labels and saves it as image_file in the working directory.
        If the 2D coordinates are not calculated yet, this method calls self._compute_2d_coords().
        """
        if not self.fit2d:
            self._compute_2d_coords()
            
        if custom_label is None:
            node_label = self._get_node_labels(start)
        else:
            node_label = custom_label

        view = rdMolDraw2D.MolDraw2DSVG(height,width)
        tm = rdMolDraw2D.PrepareMolForDrawing(self.mol)
        view.SetFontSize(0.9*view.FontSize())
        option = view.drawOptions()
        for k, v in sorted(node_label.items()):
            option.atomLabels[k] = v
        #option.atomLabels[6] = 'N1'
        #option.atomLabels[8] = 'N2'
        #option.multipleBondOffset=0.07
        #option.padding=0.11
        #option.legendFontSize=20
        view.DrawMolecule(tm, highlightAtoms=highlightAtoms, highlightAtomColors=highlightAtomColors,
                  highlightAtomRadii=highlightAtomRadii)
        view.FinishDrawing()
        svg = view.GetDrawingText()
        with open(image_file, 'w') as f:
            f.write(svg)
        return SVG(svg.replace('svg:', ''))

        #pos = self._get_coordinates()
        #node_color = self._get_node_colors()
        #if custom_label is None:
        #    node_label = self._get_node_labels(start)
        #else:
        #    node_label = custom_label
        #edge_labels = dict()
        #for edge in self.graph.edges(data=True):
        #    edge_labels[(edge[0], edge[1])] = edge[2]["index"]
        #nx.draw_networkx(self.graph, pos, node_color=node_color, alpha=0.4)
        #nx.draw_networkx_labels(self.graph, pos, fontsize=6, labels=node_label)
        #nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels,
        #                             fontsize=3, font_color="b")
        #plt.draw()

        #return True

    def _get_node_colors(self):
        """
        obtains the list of the atomic element symbols in self.graph.nodes,
        sorts the listをsort，gives the 0-indexed integer IDs for the symbols without redundancy,
        and returns the correspondence list between the integer IDs and the symbol_list.
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
        returns dict to show the label of the node.
        must be a dict for the use of NetworkX.
        This method is bacially for solving the 0-index and 1-index problem in the molblock.
        """
        d_label = {}
        l_nodes = self.graph.nodes(data=True)
        for node in l_nodes:
            d_label[node[0]] = str(int(node[1]["row"][12]) + start - 1)

        return d_label

    def find_seq(self, length, bidirectonal=True):
        """
        yields the substructures in self.graph for which the size is smaller than the one defined by length
        and returns the list of the node lists as an iterator.
        """
        for i in self.graph.nodes():
            for j in self.graph.nodes():
                if length == 3 and i == j:
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
        returns True if atom_1 and atom_2 forms a chemical bond, otherwise returns False.
        """
        return atom_1 in self.graph.adj[atom_2].keys()

    def get_symbol(self, atom_index):
        """
        returns the atomic element symbol of the corresponding atom specified by the atom_index.
        """
        return self.graph.node[atom_index]['symbol']

    def get_triplets(self):
        """
        returns all the substructures that consists of three nodes (the atoms that are not hydrogen atoms).
        """
        return self.find_seq(3, bidirectonal=True)

    def get_vicinities(self):
        """
        returns all the substructures that consists of a center atom and the attaching atoms as the data strucure as:
        [(0, [1, 2, 3]),
        (1, [3, 4, 5]),
        (2, [3, 4, 5]),
        ...
        ]
        """
        return [(j, [i for i in nx.all_neighbors(self.graph, j)])
                for j in self.graph.nodes()]
