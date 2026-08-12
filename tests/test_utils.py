import unittest

import numpy as np

from spliceai.utils import is_valid_allele, normalise_chrom, one_hot_encode


class TestIsValidAllele(unittest.TestCase):
    def test_accepts_sequence_alleles_and_rejects_vcf_placeholders(self):
        for allele in ("ACGT", "acgt", "CGTRYSWKMBDHVN", "ryswkmbdhvn"):
            self.assertTrue(is_valid_allele(allele))
        for allele in ("", ".", "*", "<DEL>", "<INS>", "G]chr17:198982]"):
            self.assertFalse(is_valid_allele(allele))


class TestNormaliseChrom(unittest.TestCase):
    def test_matches_target_prefix_style(self):
        self.assertEqual(normalise_chrom("chr10", "10"), "10")
        self.assertEqual(normalise_chrom("10", "chr10"), "chr10")
        self.assertEqual(normalise_chrom("chr10", "chr1"), "chr10")
        self.assertEqual(normalise_chrom("10", "1"), "10")


class TestOneHotEncode(unittest.TestCase):
    def test_encodes_canonical_and_ambiguous_nucleotides(self):
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
        sequence = "NnRYSWKMBDHVryswkmbdhvX?-"

        np.testing.assert_array_equal(
            one_hot_encode(sequence),
            np.zeros((len(sequence), 4), dtype=np.float32),
        )
        encoded = one_hot_encode("")

        self.assertEqual(encoded.shape, (0, 4))
        self.assertEqual(encoded.dtype, np.dtype(np.float32))


if __name__ == "__main__":
    unittest.main()
