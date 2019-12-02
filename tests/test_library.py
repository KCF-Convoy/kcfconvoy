# coding: utf-8
import os
import shutil
import unittest

from kcfconvoy import Library
from rdkit import Chem

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


class TestLibrary(unittest.TestCase):
    """
    kcfconvoyのLibraryのテスト
    """

    def setUp(self):
        shutil.rmtree("./knapsack", ignore_errors=True)
        shutil.rmtree("./kegg", ignore_errors=True)
        with open(PATH, "w") as f:
            f.write(MOLBLOCK)

    def test_input_from_kegg(self):
        """
        input_from_keggのテスト
        内部でinput_molfileを使用
        """
        cid = "C00002"
        expected = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7),
                    (3, 8), (4, 9), (4, 8), (5, 10), (6, 11), (7, 12), (7, 13),
                    (9, 14), (9, 15), (10, 14), (11, 16), (11, 12), (12, 17),
                    (16, 18), (18, 19), (19, 20), (19, 21), (19, 22), (20, 23),
                    (23, 24), (23, 25), (23, 26), (24, 27), (27, 28), (27, 29),
                    (27, 30)]
        lib = Library()
        lib.input_from_kegg(cid)
        actual = list(list(lib.cpds[0].graph.edges()))
        self.assertEqual(actual, expected)

    def test_input_from_knapsack(self):
        """
        input_from_knapsackのテスト
        内部でinput_molfileを使用
        """
        cid = "C00037855"
        expected = [(0, 1), (0, 5), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5),
                    (6, 7)]
        lib = Library()
        lib.input_from_knapsack(cid)
        actual = list(lib.cpds[0].graph.edges())
        self.assertEqual(actual, expected)

    def test_input_molfile(self):
        """
        input_molfileのテスト
        """

        expected = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
                    (5, 7), (7, 8)]
        lib = Library()
        lib.input_molfile("./test.mol")
        actual = list(lib.cpds[0].graph.edges())
        self.assertEqual(actual, expected)

    def test_input_inchi(self):
        """
        input_inchiのテスト
        内部でinput_rdkmolを使用
        """
        inchi = ("InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,"
                 "5,7-8,10-11H,1H2/t2-,5+/m0/s1")
        expected = [(0, 1), (0, 6), (1, 4), (1, 7), (2, 3), (2, 4),
                    (2, 8), (3, 5), (3, 9), (4, 11), (5, 10), (5, 11)]
        lib = Library()
        lib.input_inchi(inchi)
        actual = list(lib.cpds[0].graph.edges())
        self.assertEqual(actual, expected)

    def test_input_smiles(self):
        """
        input_smilesのテスト
        内部でinput_rdkmolを使用
        """
        smiles = 'O=Cc1ccc(O)cc1'
        expected = [(0, 1), (1, 2), (2, 3), (2, 8), (3, 4), (4, 5), (5, 6),
                    (5, 7), (7, 8)]
        lib = Library()
        lib.input_smiles(smiles)
        actual = list(lib.cpds[0].graph.edges())
        self.assertEqual(actual, expected)

    def test_input_rdkmol(self):
        """
        input_rdkmolのテスト
        """
        rdkmol = Chem.MolFromMolBlock(MOLBLOCK)
        expected = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
                    (5, 7), (7, 8)]
        lib = Library()
        lib.input_rdkmol(rdkmol)
        actual = list(lib.cpds[0].graph.edges())
        self.assertEqual(actual, expected)

    def test_calc_fingerprints(self):
        smiles = 'O=Cc1ccc(O)cc1'
        rdkmol = Chem.MolFromMolBlock(MOLBLOCK)
        inchi = ("InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,"
                 "5,7-8,10-11H,1H2/t2-,5+/m0/s1")
        lib = Library()
        lib.input_smiles(smiles)
        lib.input_rdkmol(rdkmol)
        lib.input_inchi(inchi)
        lib.calc_fingerprints()
        self.assertEqual(lib.fps[0], lib.fps[1])
        self.assertNotEqual(lib.fps[0], lib.fps[2])
        self.assertNotEqual(lib.fps[1], lib.fps[2])

    def tearDown(self):
        shutil.rmtree("./knapsack", ignore_errors=True)
        shutil.rmtree("./kegg", ignore_errors=True)
        os.remove(PATH)


if __name__ == "__main__":
    unittest.main()
