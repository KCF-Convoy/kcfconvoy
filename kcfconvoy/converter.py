# at kcfconvoy
from rdkit import Chem
import copy
import networkx as nx
import numpy as np
import pandas as pd

class KcfMatrix:
    def __init__(self):
        self.names = []
        self.kcfvecs = []
        self.all_strs = []
        self.all_mat = np.array([])
        self.mask_array = np.array([])
        self.mat = np.array([])
        
    def input_library(self, library):
        self.names = library.names
        self.kcfvecs = [kcf_vec(cpd) for cpd in library.cpds]
        self.all_strs = list(set([item for kcfvec in self.kcfvecs for item in kcfvec.strs]))
        self.all_mat = np.zeros((len(self.kcfvecs), len(self.all_strs)))
        for i, kcfvec in enumerate(self.kcfvecs):
            for j, _str in enumerate(kcfvec.strs):
                self.all_mat[i][self.all_strs.index(_str)] = kcfvec.counts[j]
        self.calc_kcf_mat()

    def calc_kcf_mat(self, ratio = 400):
        kcf_matT = self.all_mat.T
        min_cpd = max(len(self.all_mat) / ratio, 1)
        self.mask_array = np.array([len(np.where(a!=0)[0]) > min_cpd for a in kcf_matT])
        kcf_matT2 = kcf_matT[self.mask_array]
        self.mat = kcf_matT2.T

class KcfV:
    def __init__(self):
        self.strs = []
        self.types = []
        self.levels = []
        self.counts = []
        self.keggatom = {}
        self.n_nodes = []
        self.ring_string = []
        self.subs_string = []

    def add_str(self, _str, n_node, _type, lev):
        if _str not in self.strs:
            self.strs.append(_str)
            self.types.append(_type)
            self.levels.append(lev)
            self.counts.append(0)
            self.n_nodes.append(n_node)
        self.counts[self.strs.index(_str)] += 1
        return True
    def pandas(self):
        matrix = []
        for _str, _type, lev, count in zip(self.strs, self.types, self.levels, self.counts):
            matrix.append([_str, _type, lev, count])
        return pd.DataFrame(sorted(matrix, key=lambda x:x[3], reverse=True), columns=['str', 'type', 'level', 'count'])
    def string2seq(self):
        list1 = []
        for dict2 in self.subs_string:
            dict1 = {}
            for k, v in dict2.items():
                if v not in dict1.keys():
                    dict1[v] = []
                dict1[v].append(k)
            list1.append(dict1)
        return list1

def bundle(headpin, klabels, level, cutoff=False):
    skeldic2 = {}
    for head, length_seq in headpin.items():
        seen = []
        string = ''
        idx = 0
        previdx = 0
        prevnotseen = False
        mainchain = []
        bridges = []
        longerpatherror = False
        length_kgseq = [[length, [klabels[i][level] for i in seq], seq] for (length, seq) in length_seq]
        for length, seqc, seq in sorted(length_kgseq):
            if seq[0] != seq[-1]:
                skeldic2[",".join([str(x) for x in sorted(seq)])] = "-".join(seqc)
            idx += 1
            if cutoff:
                if idx == 1:
                    if length != (0 - cutoff):
                        break
            for i, atm in enumerate(seq):
                if atm in seen:
                    if i != 0:
                        if prevnotseen:
                            string += "-" + str(seen.index(atm) + 1)
                            mainchain.append(sorted([seen.index(seq[i - 1]) + 1, seen.index(atm) + 1]))
                            break
                        else:
                            bridge = sorted([seen.index(seq[i - 1]) + 1, seen.index(atm) + 1])
                            if bridge not in mainchain:
                                if bridge not in bridges:
                                    bridges.append(bridge)
                    prevnotseen = False
                else:
                    #if cutoff:
                    #    if idx != 1:
                    #        if length == 0 - cutoff:
                    #            if i < (1 - length - i):
                    #                #print(i, length, path)
                    #                #continue
                    #                #########longerpatherror = True
                    #                #########break
                    #                pass
                    if i != 0 and idx != previdx:
                        string += "," + str(seen.index(seq[i - 1]) + 1)
                    if len(string) > 0:
                        string += "-"
                    string += klabels[atm][level]
                    seen.append(atm)
                    if i != 0:
                        mainchain.append(sorted([seen.index(seq[i - 1]) + 1, seen.index(atm) + 1]))
                    previdx = idx
                    prevnotseen = True
            #if longerpatherror:
            #    break
        if len(seen) < 3:
            continue
        if len(bridges) > 0:
            string += "," + ",".join([str(b[0]) + "-" + str(b[1]) for b in sorted(bridges)])
        seen = ",".join([str(i) for i in sorted(seen)])
        if seen not in skeldic2.keys():
            skeldic2[seen] = string
        elif skeldic2[seen] > string:
            skeldic2[seen] = string
    return skeldic2

def get_coordinates(mol):
    coordinates = {}
    bonds = []
    molfile = Chem.MolToMolBlock(mol)
    n_atoms = 0
    n_bonds = 0
    for i, line in enumerate(molfile.split("\n")):
        if i < 4:
            if i == 3:
                n_atoms = int(line[:3])
                n_bonds = int(line[3:6])
        elif i < (4 + n_atoms):
            a = line.split()
            coordinates[i-4] = [a[0], a[1], a[2]]
        elif i < (4 + n_atoms + n_bonds):
            a = line.split()
            bonds.append(a)
        else:
            pass
    return coordinates, bonds

def kcf_vec(c1, levels = [0, 1, 2], attributes = [0, 1, 2, 3, 4, 5], maxsublength = 12, maxringsize = 9):
    kcfv = KcfV()
    kcfv.keggatom = rdkmol_to_keggatoms(c1.mol, data=True)
    if kcfv.keggatom == {}:
        return kcfv

    # skeleton or inorganic substructure entries
    skeleton = []
    inorganic = []
    c1c = copy.deepcopy(c1)
    for edge in c1c.graph.edges():
        ele1 = c1c.symbol(edge[0])
        ele2 = c1c.symbol(edge[1])
        if (ele1 == 'C' and ele2 != 'C') or (ele1 != 'C' and ele2 == 'C'):
            c1c.graph.remove_edge(edge[0], edge[1])
    subgraphs = nx.connected_component_subgraphs(c1c.graph)
    for subg in subgraphs:
        if len(subg.nodes()) < 4:
            continue
        if c1.symbol(subg.nodes()[0]) == 'C':
            skeleton.append(','.join([str(n) for n in sorted(subg.nodes())]))
        else:
            inorganic.append(','.join([str(n) for n in sorted(subg.nodes())]))

    pinpath1 = [pinpath(subg) for subg in subgraphs]
    pinpath2 = [[[cutoff, pinpath(subg, cutoff)] for subg in subgraphs]
                for cutoff in range(4, maxsublength + 1)]
    pinpath3 = [[cutoff, pinpath(c1.graph, cutoff)] for cutoff in range(4, maxsublength + 1)]
    
    ########################
    #print("pinpath1")
    #print(pinpath1)
    #print("pinpath2")
    #print(pinpath2)
    #print("pinpath3")
    #for cutoff, pinpath3x in pinpath3:
    #    print(cutoff)
    #    for k in pinpath3x.items():
    #        print(k)
    #    print("\n")
    ########################
    
    keggatom_levels = ['atom_species', 'atom_class', 'kegg_atom']
    for lev in [keggatom_levels[level] for level in levels]:
        if 0 in attributes:
            for atom in kcfv.keggatom.values():
                kcfv.add_str(atom[lev], 1, 'atom', lev)

        if 1 in attributes:
            for bond in c1.graph.edges():
                s = ("-").join(sorted([kcfv.keggatom[i][lev] for i in bond]))
                kcfv.add_str(s, 2, 'bond', lev)

        if 2 in attributes:
            for triplet in c1.triplets():
                s1 = ("-").join([kcfv.keggatom[i][lev] for i in triplet])
                s2 = ("-").join(reversed([kcfv.keggatom[i][lev] for i in triplet]))
                s = sorted([s1, s2])[0]
                kcfv.add_str(s, 3, 'triplet', lev)

        if 3 in attributes:
            for vicinity in c1.vicinities():
                if len(vicinity[1]) < 3:
                    continue
                vic_tmp = sorted([[kcfv.keggatom[i][lev], i] for i in vicinity[1]])
                s = "-".join([vic_tmp[0][0], kcfv.keggatom[vicinity[0]][lev],  vic_tmp[1][0]])
                for t in range(2,len(vic_tmp)):
                    s += ',2-' + vic_tmp[t][0]
                kcfv.add_str(s, 1 + len(vicinity[1]), 'vicinity', lev)
            #print(vicinity, keggatom1[vicinity[0]][lev], vic_tmp, s)

        #ring_string = {}
        kcfv.ring_string.append({})
        if 4 in attributes:
            for ring in c1.find_seq(maxringsize + 1):
                if len(ring) < 4: continue
                if ring[0] != ring[-1]: continue
                k = ','.join([str(n) for n in sorted(ring[0:-1])])
                s = '-'.join([kcfv.keggatom[n][lev] for n in ring[0:-1]])
                for i in range(len(ring) - 2):
                    for j in range(i + 2, len(ring) - 1):
                        if c1.has_bond(ring[i], ring[j]):
                            s += ',' + str(i + 1) + '-' + str(j + 1)
                if k not in kcfv.ring_string[len(kcfv.ring_string) - 1].keys():
                    kcfv.ring_string[len(kcfv.ring_string) - 1][k] = s
                elif kcfv.ring_string[len(kcfv.ring_string) - 1][k] > s:
                    kcfv.ring_string[len(kcfv.ring_string) - 1][k] = s
            for k, s in kcfv.ring_string[len(kcfv.ring_string) - 1].items():
                kcfv.add_str(s, len(k.split(',')), 'ring', lev)

        kcfv.subs_string.append({})
        if 5 in attributes:
            for pinpath1x in pinpath1:
                for k, s in bundle(pinpath1x, kcfv.keggatom, lev).items():
                    #if k in ring_basis:
                    #    s = modify_simple_ring(k, s)
                    if k not in subs_string[len(kcfv.subs_string) - 1].keys():
                        subs_string[len(kcfv.subs_string) - 1][k] = s
                    elif subs_string[len(kcfv.subs_string) - 1][k] > s:
                        subs_string[len(kcfv.subs_string) - 1][k] = s
            for pinpath2x in pinpath2:
                for cutoff, pinpath2y in pinpath2x:
                    for k, s in bundle(pinpath2y, kcfv.keggatom, lev, cutoff).items():
                        #if k in ring_basis:
                        #    s = modify_simple_ring(k, s)
                        if k not in subs_string[len(kcfv.subs_string) - 1].keys():
                            subs_string[len(kcfv.subs_string) - 1][k] = s
                        elif subs_string[len(kcfv.subs_string) - 1][k] > s:
                            subs_string[len(kcfv.subs_string) - 1][k] = s
            for cutoff, pinpath3x in pinpath3:
                #print("cutoff=", cutoff, " pinpath3x=", pinpath3x) ###############
                for k, s in bundle(pinpath3x, kcfv.keggatom, lev, cutoff).items():
                    #print("bundle", k, s) ####################
                    #if k in ring_basis:
                    #    s = modify_simple_ring(k, s)
                    if k not in kcfv.subs_string[len(kcfv.subs_string) - 1].keys():
                        kcfv.subs_string[len(kcfv.subs_string) - 1][k] = s
                    elif kcfv.subs_string[len(kcfv.subs_string) - 1][k] > s:
                        kcfv.subs_string[len(kcfv.subs_string) - 1][k] = s

            for k, s in kcfv.subs_string[len(kcfv.subs_string) - 1].items():
                if k in kcfv.ring_string[len(kcfv.subs_string) - 1].keys():
                    continue
                elif k in skeleton:
                    kcfv.add_str(s, len(k.split(',')), 'skeleton', lev)
                elif k in inorganic:
                    kcfv.add_str(s, len(k.split(',')), 'inorganic', lev)
                elif len(s.split(',')) == 1:
                    kcfv.add_str(s, len(k.split(',')), 'linear', lev)
                else:
                    kcfv.add_str(s, len(k.split(',')), 'unit', lev)

    return kcfv

def pinpath(graph, cutoff=False):
    headpin = {}
    for i in graph.nodes():
        for j in graph.nodes():
            sequences = []
            if cutoff:
                 for seq in nx.all_simple_paths(graph, i, j, cutoff=cutoff):
                        if len(seq) > cutoff:
                            if seq[0] != seq[-1]:
                                continue
                        sequences.append(seq)
            else:
                sequences = [seq for seq in nx.all_simple_paths(graph, i, j)]
            for seq in sequences:
                if (seq[0] == seq[-1]) and (len(seq) == 3):
                    continue
                head = "-".join([str(seq[0]), str(seq[1])])
                if head not in headpin.keys():
                    headpin[head] = []
                length = len(seq)
                if i == j:
                    length -= 1
                headpin[head].append([0 - length, seq])
    return headpin

def rdkmol_to_kcf(mol, id='NoName'):
    lines = ''
    labels = rdkmol_to_keggatoms(mol, data=True)
    lines += "ENTRY       %-28s  Compound\n" % (id)
    lines += "ATOM        %-5d\n" % (mol.GetNumAtoms())
    atoms, bonds = get_coordinates(mol)
    for atom in sorted(atoms.items()):
        lines += "            "
        lines += "%-3d %-3s %-2s  %8s  %8s\n" % (atom[0] + 1, labels[atom[0]]['kegg_atom'],
                                                 labels[atom[0]]['atom_species'], atom[1][0], atom[1][1])
    lines += "BOND        %-5d\n" % (mol.GetNumBonds())
    for idx, bond in enumerate(bonds):
        lines += "            "
        lines += "%-3d%4s %3s %1s\n" % (idx + 1, bond[0], bond[1], bond[2])
    lines += "///\n"
    return lines

def rdkmol_to_keggatoms(m, data=False):
    result = {}
    if m == None:
        return result
    for atom in m.GetAtoms():
        if atom.GetSymbol() == 'C':
            Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
            is_in_ring = False
            if atom.IsInRing():
                if ring_size_checker(atom):
                    is_in_ring = True
            if atom.GetIsAromatic():
                if Explicit == 3:
                    result[atom.GetIdx()] = set_keggatom('C8x')
                elif Explicit == 4:
                    result[atom.GetIdx()] = set_keggatom('C8y')
                else:
                    result[atom.GetIdx()] = set_keggatom('C0')
            elif Explicit == len(atom.GetNeighbors()):
                if is_in_ring:
                    if Explicit == 2:
                        result[atom.GetIdx()] = set_keggatom('C1x')
                    elif Explicit == 3:
                        result[atom.GetIdx()] = set_keggatom('C1y')
                    elif Explicit == 4:
                        result[atom.GetIdx()] = set_keggatom('C1z')
                    else:
                        result[atom.GetIdx()] = set_keggatom('C0')
                else:
                    if Explicit == 1:
                        result[atom.GetIdx()] = set_keggatom('C1a')
                    elif Explicit == 2:
                        result[atom.GetIdx()] = set_keggatom('C1b')
                    elif Explicit == 3:
                        result[atom.GetIdx()] = set_keggatom('C1c')
                    elif Explicit == 4:
                        result[atom.GetIdx()] = set_keggatom('C1d')
                    else:
                        result[atom.GetIdx()] = set_keggatom('C0')
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
                                is_ester =True

                    if is_carbonyl:
                        if num_neighbor_oxygen_atoms == 1:
                            if Explicit == 3:
                                result[atom.GetIdx()] = set_keggatom('C4a')
                            elif Explicit == 4:
                                if is_in_ring:
                                    result[atom.GetIdx()] = set_keggatom('C5x')
                                else:
                                    result[atom.GetIdx()] = set_keggatom('C5a')
                            elif Explicit == 2:
                                # for formaldehyde
                                result[atom.GetIdx()] = set_keggatom('C2a')
                            else:
                                result[atom.GetIdx()] = set_keggatom('C0')
                        elif num_neighbor_oxygen_atoms <= 3:
                            if is_ester:
                                if is_in_ring:
                                    result[atom.GetIdx()] = set_keggatom('C7x')
                                else:
                                    result[atom.GetIdx()] = set_keggatom('C7a')
                            else:
                                result[atom.GetIdx()] = set_keggatom('C6a')
                        else:
                            result[atom.GetIdx()] = set_keggatom('C0')
                    else:
                        if is_in_ring:
                            if Explicit == 3:
                                result[atom.GetIdx()] = set_keggatom('C2x')
                            elif Explicit == 4:
                                result[atom.GetIdx()] = set_keggatom('C2y')
                            else:
                                result[atom.GetIdx()] = set_keggatom('C0')
                        else:
                            if Explicit == 2:
                                result[atom.GetIdx()] = set_keggatom('C2a')
                            elif Explicit == 3:
                                result[atom.GetIdx()] = set_keggatom('C2b')
                            elif Explicit == 4:
                                result[atom.GetIdx()] = set_keggatom('C2c')
                            else:
                                result[atom.GetIdx()] = set_keggatom('C0')
            elif Explicit - len(atom.GetNeighbors()) == 2:
                if Explicit == 3:
                    result[atom.GetIdx()] = set_keggatom('C3a')
                elif Explicit == 4:
                    num_double = 0
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            num_double += 1
                    if num_double == 2:
                        result[atom.GetIdx()] = set_keggatom('C0')
                    elif is_in_ring:
                        result[atom.GetIdx()] = set_keggatom('C0')
                    else:
                        result[atom.GetIdx()] = set_keggatom('C3b')
                else:
                    result[atom.GetIdx()] = set_keggatom('C0')
            else:
                result[atom.GetIdx()] = set_keggatom('C0')
        elif atom.GetSymbol() == 'N':
            Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
            is_in_ring = False
            if atom.IsInRing():
                if ring_size_checker(atom):
                    is_in_ring = True
            if atom.GetIsAromatic():
                if len(atom.GetNeighbors()) == 2:
                    if Explicit == 2:
                        result[atom.GetIdx()] = set_keggatom('N4x')
                    elif Explicit == 3:
                        result[atom.GetIdx()] = set_keggatom('N5x')
                    else:
                        result[atom.GetIdx()] = set_keggatom('N0')
                elif len(atom.GetNeighbors()) == 3:
                    if Explicit == 3:
                        result[atom.GetIdx()] = set_keggatom('N4y')
                    elif Explicit == 4:
                        result[atom.GetIdx()] = set_keggatom('N5y')
                    else:
                        result[atom.GetIdx()] = set_keggatom('N0')
                else:
                    result[atom.GetIdx()] = set_keggatom('N0')
            elif Explicit == len(atom.GetNeighbors()):
                if is_in_ring:
                    if Explicit == 2:
                        result[atom.GetIdx()] = set_keggatom('N1x')
                    elif Explicit == 3:
                        result[atom.GetIdx()] = set_keggatom('N1y')
                    elif Explicit == 4:
                        # for C06163 and others
                        result[atom.GetIdx()] = set_keggatom('N2y')
                    else:
                        result[atom.GetIdx()] = set_keggatom('N0')
                elif Explicit == 1:
                    result[atom.GetIdx()] = set_keggatom('N1a')
                elif Explicit == 2:
                    result[atom.GetIdx()] = set_keggatom('N1b')
                elif Explicit == 3:
                    result[atom.GetIdx()] = set_keggatom('N1c')
                elif Explicit == 4:
                    result[atom.GetIdx()] = set_keggatom('N1d')
                else:
                    result[atom.GetIdx()] = set_keggatom('N0')
            elif Explicit - len(atom.GetNeighbors()) == 1:
                if is_in_ring:
                    if Explicit == 3:
                        result[atom.GetIdx()] = set_keggatom('N2x')
                    elif Explicit == 4:
                        result[atom.GetIdx()] = set_keggatom('N2y')
                    else:
                        result[atom.GetIdx()] = set_keggatom('N0')
                elif Explicit == 2:
                    """
                    # for C11282
                    # occured other errors
                    if atom.GetFormalCharge() == -1:
                        # for azidopine
                        result[atom.GetIdx()] = set_keggatom('N3a')
                    else:
                    """
                    result[atom.GetIdx()] = set_keggatom('N2a')
                elif Explicit == 3:
                    result[atom.GetIdx()] = set_keggatom('N2b')
                elif Explicit <= 5:
                    # for Nitrate
                    result[atom.GetIdx()] = set_keggatom('N2b')
                else:
                    result[atom.GetIdx()] = set_keggatom('N0')
            elif Explicit - len(atom.GetNeighbors()) == 2:
                if len(atom.GetNeighbors()) == 1:
                    result[atom.GetIdx()] = set_keggatom('N3a')
                elif atom.GetFormalCharge() == 1:
                    nitrous = True
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                            nitrous = False
                    if nitrous:
                        result[atom.GetIdx()] = set_keggatom('N0')
                    else:
                        result[atom.GetIdx()] = set_keggatom('N3a')
                else:
                    result[atom.GetIdx()] = set_keggatom('N0')
            else:
                result[atom.GetIdx()] = set_keggatom('N0')
        elif atom.GetSymbol() == 'O':
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
            is_in_ring = False
            Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
            if atom.IsInRing():
                if ring_size_checker(atom):
                    is_in_ring = True

            for bond in atom.GetBonds():
                if bond.GetEndAtom().GetSymbol() != "O":
                    bond_partner = bond.GetEndAtom()
                else:
                    bond_partner = bond.GetBeginAtom()
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    has_double = True

                if bond_partner.GetSymbol() == "C":
                    if bond_partner.IsInRing():
                        if ring_size_checker(bond_partner):
                            neighbor_carbon_is_in_ring = True
                    if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        if len(bond_partner.GetBonds()) == 2:
                            if bond_partner.GetExplicitValence() - bond_partner.GetNumExplicitHs() == 4:
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
                            if neighbor2.GetExplicitValence() - neighbor2.GetNumExplicitHs() == 1:
                                is_phosphric = True
                        else:
                            num_p_r += 1
                    if num_p_r + 4 - len(bond_partner.GetNeighbors()) >= 2:
                        is_phosphric = False
                elif bond_partner.GetSymbol() == "S":
                    num_neighbor_sulfur_atoms += 1
                    for neighbor2 in bond_partner.GetNeighbors():
                        if neighbor2.GetSymbol() == "O":
                            if neighbor2.GetExplicitValence() - neighbor2.GetNumExplicitHs() == 1:
                                is_sulfuric = True

            if num_neighbor_carbonyl_carbons == 0:
                if Explicit == 1:
                    if num_neighbor_nitrogen_atoms == 1:
                        if is_nitro:
                            result[atom.GetIdx()] = set_keggatom('O3a')
                        else:
                            result[atom.GetIdx()] = set_keggatom('O1b')
                    elif num_neighbor_phosphorus_atoms == 1:
                        result[atom.GetIdx()] = set_keggatom('O1c')
                    elif num_neighbor_sulfur_atoms == 1:
                        result[atom.GetIdx()] = set_keggatom('O1d')
                    else:
                        result[atom.GetIdx()] = set_keggatom('O1a')
                elif Explicit == 2:
                    if has_double:
                        if num_neighbor_nitrogen_atoms == 1:
                            result[atom.GetIdx()] = set_keggatom('O3a')
                        elif num_neighbor_phosphorus_atoms == 1:
                            if is_phosphric:
                                result[atom.GetIdx()] = set_keggatom('O1c')
                            else:
                                result[atom.GetIdx()] = set_keggatom('O3b')
                        elif num_neighbor_sulfur_atoms == 1:
                            if is_sulfuric:
                                result[atom.GetIdx()] = set_keggatom('O1d')
                            else:
                                result[atom.GetIdx()] = set_keggatom('O3c')
                        else:
                            result[atom.GetIdx()] = set_keggatom('O0')
                    else:
                        if is_in_ring:
                            result[atom.GetIdx()] = set_keggatom('O2x')
                        elif num_neighbor_phosphorus_atoms == 0:
                            result[atom.GetIdx()] = set_keggatom('O2a')
                        elif num_neighbor_phosphorus_atoms == 1:
                            result[atom.GetIdx()] = set_keggatom('O2b')
                        elif num_neighbor_phosphorus_atoms == 2:
                            result[atom.GetIdx()] = set_keggatom('O2c')
                        else:
                            result[atom.GetIdx()] = set_keggatom('O0')
                else:
                    result[atom.GetIdx()] = set_keggatom('O0')
            elif num_neighbor_carbonyl_carbons <= 2:
                if is_co2:
                    result[atom.GetIdx()] = set_keggatom('O0')
                elif num_neighbor_carbons_neighbor_oxygens == 1:
                    if neighbor_carbon_is_in_ring:
                        result[atom.GetIdx()] = set_keggatom('O5x')
                    elif is_formaldehyde:
                        result[atom.GetIdx()] = set_keggatom('O0')
                    elif is_aldehyde:
                        result[atom.GetIdx()] = set_keggatom('O4a')
                    else:
                        result[atom.GetIdx()] = set_keggatom('O5a')
                elif 2 <= num_neighbor_carbons_neighbor_oxygens <= 4:
                    if has_double:
                        result[atom.GetIdx()] = set_keggatom('O6a')
                    elif Explicit == 1:
                        result[atom.GetIdx()] = set_keggatom('O6a')
                    elif Explicit == 2:
                        if is_in_ring:
                            result[atom.GetIdx()] = set_keggatom('O7x')
                        else:
                            result[atom.GetIdx()] = set_keggatom('O7a')
                    else:
                        result[atom.GetIdx()] = set_keggatom('O0')
                else:
                    result[atom.GetIdx()] = set_keggatom('O0')
            else:
                result[atom.GetIdx()] = set_keggatom('O0')

        elif atom.GetSymbol() == 'S':
            num_neighbor_oxygen_atoms = 0
            num_neighbor_sulfur_atoms = 0
            Explicit = atom.GetExplicitValence() - atom.GetNumExplicitHs()
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    num_neighbor_oxygen_atoms += 1
                elif neighbor.GetSymbol() == 'S':
                    num_neighbor_sulfur_atoms += 1
            if atom.IsInRing():
                if num_neighbor_sulfur_atoms > 0:
                    result[atom.GetIdx()] = set_keggatom('S3x')
                else:
                    result[atom.GetIdx()] = set_keggatom('S2x')
            elif Explicit == 1:
                result[atom.GetIdx()] = set_keggatom('S1a')
            elif num_neighbor_oxygen_atoms > 0:
                result[atom.GetIdx()] = set_keggatom('S4a')
            elif num_neighbor_sulfur_atoms > 0:
                result[atom.GetIdx()] = set_keggatom('S3a')
            elif len(atom.GetNeighbors()) == 2:
                result[atom.GetIdx()] = set_keggatom('S2a')
            else:
                result[atom.GetIdx()] = set_keggatom('S0')
        elif atom.GetSymbol() == 'P':
            num_neighbor_oxygen_atoms = 0
            num_neighbor_sulfur_atoms = 0
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    num_neighbor_oxygen_atoms += 1
                elif neighbor.GetSymbol() == "S":
                    num_neighbor_sulfur_atoms += 1
            if num_neighbor_sulfur_atoms == 0 and num_neighbor_oxygen_atoms >= 3:
                result[atom.GetIdx()] = set_keggatom('P1b')
            else:
                result[atom.GetIdx()] = set_keggatom('P1a')
        elif atom.GetSymbol() == 'F':
            result[atom.GetIdx()] = set_keggatom('X', atom)
        elif atom.GetSymbol() == 'Cl':
            result[atom.GetIdx()] = set_keggatom('X', atom)
        elif atom.GetSymbol() == 'Br':
            result[atom.GetIdx()] = set_keggatom('X', atom)
        elif atom.GetSymbol() == 'I':
            result[atom.GetIdx()] = set_keggatom('X', atom)
        elif atom.GetSymbol() == "*":
            result[atom.GetIdx()] = set_keggatom("R")
        else:
            result[atom.GetIdx()] = set_keggatom('Z', atom)
    if data:
        return result
    else:
        return {k:v['kegg_atom'] for k, v in result.items()}

def ring_size_checker(atom):
    for i in range(3, 21):
        if atom.IsInRingSize(i):
            return True
    return False

def set_keggatom(str, atom=None):
    if atom is None:
        return {'atom_species': str[0], 'atom_class': str[0:2], 'kegg_atom': str}
    else:
        if atom.GetSymbol()[0] != "R":
            return {"atom_species": atom.GetSymbol(), "atom_class": str, "kegg_atom": str}
        else:
            return {"atom_species": "R#", "atom_class": str, "kegg_atom": str}

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
