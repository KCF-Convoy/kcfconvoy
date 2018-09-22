# coding: utf-8
import unittest
from kcfconvoy import KCFvec
import os
from rdkit import Chem
import pandas as pd
from collections import defaultdict

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


class TestKCFvec(unittest.TestCase):
    """
    kcfconvoyのKCFvecのテスト
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
        expected = KEGG_ATOM_LABEL
        vec = KCFvec()
        vec.input_from_kegg(cid)
        actual = vec.kegg_atom_label
        self.assertEqual(actual, expected)

    def test_input_from_knapsack(self):
        """
        input_from_knapsackのテスト
        この場合KEGG_ATOM_LABELと順番が違うので空のdict()でないことのみ確認している
        """
        cid = "C00002657"
        nexpected = dict()
        vec = KCFvec()
        vec.input_from_knapsack(cid)
        actual = vec.kegg_atom_label
        self.assertNotEqual(actual, nexpected)

    def test_input_molfile(self):
        """
        input_from_molfileのテスト
        """
        expected = KEGG_ATOM_LABEL
        vec = KCFvec()
        vec.input_molfile(PATH)
        actual = vec.kegg_atom_label
        self.assertEqual(actual, expected)

    def test_input_inchi(self):
        """
        input_from_molfileのテスト
        この場合KEGG_ATOM_LABELと順番が違うので空のdict()でないことのみ確認している
        """
        inchi = "InChI=1S/C7H6O2/c8-5-6-1-3-7(9)4-2-6/h1-5,9H"
        nexpected = dict()
        vec = KCFvec()
        vec.input_inchi(inchi)
        actual = vec.kegg_atom_label
        self.assertNotEqual(actual, nexpected)

    def test_input_smiles(self):
        """
        input_smilesのテスト
        この場合KEGG_ATOM_LABELと順番が違うので空のdict()でないことのみ確認している
        """
        smiles = 'O=Cc1ccc(O)cc1'
        nexpected = dict()
        vec = KCFvec()
        vec.input_smiles(smiles)
        actual = vec.kegg_atom_label
        self.assertNotEqual(actual, nexpected)

    def test_input_rdkmol(self):
        """
        input_rdkmolのテスト
        """
        rdkmol = Chem.MolFromMolBlock(MOLBLOCK)
        expected = KEGG_ATOM_LABEL
        vec = KCFvec()
        vec.input_rdkmol(rdkmol)
        actual = vec.kegg_atom_label
        self.assertEqual(actual, expected)

    def test_convert_kcf_vec(self):
        """
        convert_kcf_vecのテスト
        """
        vec = KCFvec()
        vec.input_molfile(PATH)
        vec.convert_kcf_vec()
        actual = vec.kcf_vec
        self.assertIsInstance(actual, defaultdict)

    def test_get_pandas_df(self):
        """
        get_pandas_dfのテスト
        """
        vec = KCFvec()
        vec.input_molfile(PATH)
        vec.convert_kcf_vec()
        df = vec.get_pandas_df()
        self.assertIsInstance(df, pd.DataFrame)

    def test_string2seq(self):
        """
        string2seqのテスト
        """
        expected = []
        vec = KCFvec()
        vec.input_molfile(PATH)
        vec.convert_kcf_vec()
        actual = vec.string2seq
        self.assertNotEqual(actual, expected)

    @classmethod
    def tearDownClass(cls):
        os.remove(PATH)


if __name__ == "__main__":
    unittest.main()
