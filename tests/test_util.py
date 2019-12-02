# coding: utf-8
import unittest

from kcfconvoy import KCFvec, similarity
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


class TestUtil(unittest.TestCase):
    """
    kcfconvoyのutilのテスト
    """

    def test_similarity(self):
        """
        similarityのテスト
        """
        rdkmol = Chem.MolFromMolBlock(MOLBLOCK)
        expected = (1.0, 1.0, 1.0)
        vec = KCFvec()
        vec.input_rdkmol(rdkmol)
        vec.convert_kcf_vec()
        actual = similarity(vec, vec)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
