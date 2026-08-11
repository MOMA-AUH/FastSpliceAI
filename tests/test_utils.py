import unittest

import numpy as np

from spliceai.utils import is_valid_allele, normalise_chrom, one_hot_encode


class TestIsValidAllele(unittest.TestCase):
    def test_canonical_bases(self):
        bases = list("ACGTacgt")
        for base in bases:
            self.assertTrue(is_valid_allele(base))

    def test_ambiguous_bases(self):
        bases = list("CGTRYSWKMBDHVNryswkmbdhvn")
        for base in bases:
            self.assertTrue(is_valid_allele(base))

    def test_no_allele(self):
        self.assertFalse(is_valid_allele(""))
        self.assertFalse(is_valid_allele("."))

    def test_missing_sequence_context(self):
        self.assertFalse(is_valid_allele("*"))

    def test_symbolic_sv_allele_codes(self):
        alleles = [
            "<DEL>",
            "<INS>",
            "<DUP>",
            "<INV>",
            "<CNV>",
            "<BND>",
            "<TRA>",
        ]
        for allele in alleles:
            self.assertFalse(is_valid_allele(allele))

    def test_bnd_allele(self):
        self.assertFalse(is_valid_allele("G]chr17:198982]"))


class TestNormaliseChrom(unittest.TestCase):
    def test_matches_target_prefix_style(self):
        self.assertEqual(normalise_chrom("chr10", "10"), "10")
        self.assertEqual(normalise_chrom("10", "chr10"), "chr10")
        self.assertEqual(normalise_chrom("chr10", "chr1"), "chr10")
        self.assertEqual(normalise_chrom("10", "1"), "10")


class TestOneHotEncode(unittest.TestCase):
    def test_encodes_canonical_nucleotides(self):
        expected = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        np.testing.assert_array_equal(one_hot_encode("ACGT"), expected)
        np.testing.assert_array_equal(one_hot_encode("acgt"), expected)

    def test_encodes_ambiguous_and_unknown_bases_as_zeros(self):
        sequence = "NnRYSWKMBDHVryswkmbdhvX?-"

        np.testing.assert_array_equal(
            one_hot_encode(sequence),
            np.zeros((len(sequence), 4), dtype=np.float32),
        )

    def test_empty_sequence_preserves_shape_and_dtype(self):
        encoded = one_hot_encode("")

        self.assertEqual(encoded.shape, (0, 4))
        self.assertEqual(encoded.dtype, np.dtype(np.float32))


if __name__ == "__main__":
    unittest.main()
