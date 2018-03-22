import unittest
from kcfconvoy.KCFmat import KCFmat
from kcfconvoy.KCFvec import KCFvec
import os
from rdkit import Chem
import numpy as np

KEGG_ATOM_LABEL = \
            {0: {'atom_class': 'C8', 'atom_species': 'C', 'kegg_atom': 'C8y'},
             1: {'atom_class': 'C8', 'atom_species': 'C', 'kegg_atom': 'C8x'},
             2: {'atom_class': 'C8', 'atom_species': 'C', 'kegg_atom': 'C8x'},
             3: {'atom_class': 'C4', 'atom_species': 'C', 'kegg_atom': 'C4a'},
             4: {'atom_class': 'C8', 'atom_species': 'C', 'kegg_atom': 'C8x'},
             5: {'atom_class': 'C8', 'atom_species': 'C', 'kegg_atom': 'C8x'},
             6: {'atom_class': 'O4', 'atom_species': 'O', 'kegg_atom': 'O4a'},
             7: {'atom_class': 'C8', 'atom_species': 'C', 'kegg_atom': 'C8y'},
             8: {'atom_class': 'O1', 'atom_species': 'O', 'kegg_atom': 'O1a'}}

MOLBLOCK = (
    " \n"
    " \n"
    " \n"
    "  9  9  0  0  0  0  0  0  0  0999 V2000\n"
    "   28.0314  -18.4596    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   26.8219  -19.1573    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   29.2539  -19.1573    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   28.0250  -17.0706    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   26.8219  -20.5653    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   29.2539  -20.5653    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   29.2283  -16.3732    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   28.0314  -21.2759    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   28.0250  -22.6711    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0     0  0\n"
    "  1  3  2  0     0  0\n"
    "  1  4  1  0     0  0\n"
    "  2  5  2  0     0  0\n"
    "  3  6  1  0     0  0\n"
    "  4  7  2  0     0  0\n"
    "  5  8  1  0     0  0\n"
    "  8  9  1  0     0  0\n"
    "  6  8  2  0     0  0\n"
    "M  END\n"
)
PATH = "./test.mol"


class TestKCFmat(unittest.TestCase):
    """
    kcfconvoyのKCFmatのテスト
    """

    @classmethod
    def setUpClass(cls):
        with open(PATH, "w")as f:
            f.write(MOLBLOCK)

    def test_input_from_kegg(self):
        """
        input_from_keggのテスト
        """
        cid = "C00633"
        expected = KCFvec
        mat = KCFmat()
        mat.input_from_kegg(cid)
        mat.input_from_kegg(cid)
        actual = mat.kcf_vecs
        self.assertIsInstance(actual[0], expected)
        self.assertIsInstance(actual[1], expected)
        self.assertEqual(actual[0].kegg_atom_label, KEGG_ATOM_LABEL)

    def test_input_from_knapsack(self):
        """
        input_from_knapsackのテスト
        """
        cid = "C00002657"
        expected = KCFvec
        mat = KCFmat()
        mat.input_from_knapsack(cid)
        mat.input_from_knapsack(cid)
        actual = mat.kcf_vecs
        self.assertIsInstance(actual[0], expected)
        self.assertIsInstance(actual[1], expected)

    def test_input_molfile(self):
        """
        input_from_molfileのテスト
        """
        expected = KCFvec
        mat = KCFmat()
        mat.input_molfile(PATH)
        mat.input_molfile(PATH)
        actual = mat.kcf_vecs
        self.assertIsInstance(actual[0], expected)
        self.assertIsInstance(actual[1], expected)
        self.assertEqual(actual[0].kegg_atom_label, KEGG_ATOM_LABEL)

    def test_input_inchi(self):
        """
        input_inchiのテスト
        """
        inchi = "InChI=1S/C7H6O2/c8-5-6-1-3-7(9)4-2-6/h1-5,9H"
        expected = KCFvec
        mat = KCFmat()
        mat.input_inchi(inchi)
        mat.input_inchi(inchi)
        actual = mat.kcf_vecs
        self.assertIsInstance(actual[0], expected)
        self.assertIsInstance(actual[1], expected)

    def test_input_smiles(self):
        """
        input_smilesのテスト
        """
        smiles = 'O=Cc1ccc(O)cc1'
        expected = KCFvec
        mat = KCFmat()
        mat.input_smiles(smiles)
        mat.input_smiles(smiles)
        actual = mat.kcf_vecs
        self.assertIsInstance(actual[0], expected)
        self.assertIsInstance(actual[1], expected)

    def test_input_rdkmol(self):
        """
        input_rdkmolのテスト
        """
        rdkmol = Chem.MolFromMolBlock(MOLBLOCK)
        expected = KCFvec
        mat = KCFmat()
        mat.input_rdkmol(rdkmol)
        mat.input_rdkmol(rdkmol)
        actual = mat.kcf_vecs
        self.assertIsInstance(actual[0], expected)
        self.assertIsInstance(actual[1], expected)
        self.assertEqual(actual[0].kegg_atom_label, KEGG_ATOM_LABEL)

    def test_calc_kcf_matrix(self):
        """
        calc_kcf_matrixのテスト
        """
        nexpected = np.array([])
        mat = KCFmat()
        mat.input_molfile(PATH)
        mat.input_molfile(PATH)
        mat.calc_kcf_matrix()
        actual1 = mat.all_mat
        actual2 = mat.mat
        self.assertNotEqual(actual1, nexpected)
        self.assertNotEqual(actual2, nexpected)

    @classmethod
    def tearDownClass(cls):
        os.remove(PATH)


if __name__ == "__main__":
    unittest.main()
