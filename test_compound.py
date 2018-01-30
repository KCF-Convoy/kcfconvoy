import unittest
from kcfconvoy.Compound import Compound


class TestCompound(unittest.TestCase):
    """
    kcfconvoyのCompoundのテスト
    """

    def test_input_from_kegg(self):
        """
        input_from_keggのテスト
        内部でinput_molfileを使用
        作成したmolファイルとkeggフォルダを消すか検討中
        """
        cid = "C00002"
        expected = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7),
                    (3, 8), (4, 8), (4, 9), (5, 10), (6, 11), (7, 12), (7, 13),
                    (9, 14), (9, 15), (10, 14), (11, 16), (11, 12), (12, 17),
                    (16, 18), (18, 19), (19, 20), (19, 21), (19, 22), (20, 23),
                    (23, 24), (23, 25), (23, 26), (24, 27), (27, 28), (27, 29),
                    (27, 30)]
        c = Compound()
        c.input_from_kegg(cid)
        actual = c.graph.edges()
        self.assertEqual(actual, expected)

    def test_input_from_knapsack(self):
        """
        input_from_knapsackのテスト
        内部でinput_molfileを使用
        作成したmolファイルとknapsackフォルダを消すか検討中
        """
        cid = "C00037855"
        expected = [(0, 1), (0, 5), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5),
                    (6, 7)]
        c = Compound()
        c.input_from_knapsack(cid)
        actual = c.graph.edges()
        self.assertEqual(actual, expected)

    def test_input_molfile(self):
        """
        input_molfileのテスト
        """
        path = "test.mol"
        expected = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
                    (5, 7), (7, 8)]
        c = Compound()
        c.input_molfile(path)
        actual = c.graph.edges()
        self.assertEqual(actual, expected)

    def test_input_inchi(self):
        """
        input_inchiのテスト
        内部でinput_rdkmolを使用
        """
        inchi = ("InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,"
                 "5,7-8,10-11H,1H2/t2-,5+/m0/s1")
        expected = [(0, 1), (0, 6), (1, 4), (1, 7), (2, 8), (2, 3), (2, 4),
                    (3, 9), (3, 5), (4, 11), (5, 11), (5, 10)]
        c = Compound()
        c.input_inchi(inchi)
        actual = c.graph.edges()
        self.assertEqual(actual, expected)

    def test_input_smiles(self):
        """
        input_smilesのテスト
        内部でinput_rdkmolを使用
        """
        smiles = 'O=Cc1ccc(O)cc1'
        expected = [(0, 1), (1, 2), (2, 8), (2, 3), (3, 4), (4, 5), (5, 6),
                    (5, 7), (7, 8)]
        c = Compound()
        c.input_smiles(smiles)
        actual = c.graph.edges()
        self.assertEqual(actual, expected)

    def test_input_rdkmol(self):
        """
        input_rdkmolのテスト
        """
        from rdkit import Chem
        path = "test.mol"
        rdkmol = Chem.MolFromMolFile(path)
        expected = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
                    (5, 7), (7, 8)]
        c = Compound()
        c.input_rdkmol(rdkmol)
        actual = c.graph.edges()
        self.assertEqual(actual, expected)

    def test_draw_cpd(self):
        """
        draw_cpdのテスト
        """
        import os
        path = "test.mol"
        c = Compound()
        c.input_molfile(path)
        c.draw_cpd("test.png")
        self.assertTrue(os.path.exists("./test.png"))
        os.remove("./test.png")

    def test_draw_cpd_with_labels(self):
        """
        draw_cpd_with_labelsのテスト
        これは実行時にエラーが出るかのチェックしかしてない
        """
        path = "test.mol"
        c = Compound()
        c.input_molfile(path)
        c.draw_cpd_with_labels()

    def test_find_seq(self):
        """
        find_seqのテスト
        """
        path = "test.mol"
        expected = [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 1, 4], [0, 2, 5],
                    [0, 3, 6], [0, 1, 4, 7], [0, 2, 5, 7], [1, 0, 1],
                    [1, 4, 1], [1, 0, 2], [1, 0, 3], [1, 0, 2, 5],
                    [1, 4, 7, 5], [1, 0, 3, 6], [1, 4, 7], [1, 4, 7, 8],
                    [2, 0, 1], [2, 0, 2], [2, 5, 2], [2, 0, 3], [2, 0, 1, 4],
                    [2, 5, 7, 4], [2, 0, 3, 6], [2, 5, 7], [2, 5, 7, 8],
                    [3, 0, 1], [3, 0, 2], [3, 0, 3], [3, 6, 3], [3, 0, 1, 4],
                    [3, 0, 2, 5], [4, 1, 0], [4, 1, 0, 2], [4, 7, 5, 2],
                    [4, 1, 0, 3], [4, 1, 4], [4, 7, 4], [4, 7, 5], [4, 7, 8],
                    [5, 2, 0], [5, 2, 0, 1], [5, 7, 4, 1], [5, 2, 0, 3],
                    [5, 7, 4], [5, 2, 5], [5, 7, 5], [5, 7, 8], [6, 3, 0],
                    [6, 3, 0, 1], [6, 3, 0, 2], [6, 3, 6], [7, 4, 1, 0],
                    [7, 5, 2, 0], [7, 4, 1], [7, 5, 2], [7, 8, 7], [7, 4, 7],
                    [7, 5, 7], [8, 7, 4, 1], [8, 7, 5, 2], [8, 7, 4],
                    [8, 7, 5], [8, 7, 8]]
        c = Compound()
        c.input_molfile(path)
        actual = [i for i in c.find_seq(4)]
        self.assertEqual(actual, expected)

    def test_has_bond(self):
        """
        has_bondのテスト
        assertをループにかけるので時間がかかるようなら要修正
        """
        path = "test.mol"
        expected = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
                    (5, 7), (7, 8)]
        c = Compound()
        c.input_molfile(path)
        for atom_1, atom_2 in expected:
            self.assertTrue(c.has_bond(atom_1, atom_2))

    def test_get_symbol(self):
        """
        get_symbolのテスト
        """
        path = "test.mol"
        expected = ["C", "C", "C", "C", "C", "C", "O", "C", "O"]
        c = Compound()
        c.input_molfile(path)
        actual = [c.get_symbol(i) for i in range(c.n_atoms)]
        self.assertEqual(actual, expected)

    def test_get_triplets(self):
        """
        get_tripletsのテスト
        内部でfind_seqを使用
        """
        path = "test.mol"
        expected = [[0, 1, 4], [0, 2, 5], [0, 3, 6], [1, 0, 2], [1, 0, 3],
                    [1, 4, 7], [2, 0, 3], [2, 5, 7], [4, 7, 5], [4, 7, 8],
                    [5, 7, 8]]
        c = Compound()
        c.input_molfile(path)
        actual = [i for i in c.get_triplets()]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
