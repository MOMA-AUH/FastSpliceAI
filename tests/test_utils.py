import unittest

from spliceai.utils import normalise_chrom


class TestNormaliseChrom(unittest.TestCase):
    def test_matches_target_prefix_style(self):
        self.assertEqual(normalise_chrom("chr10", "10"), "10")
        self.assertEqual(normalise_chrom("10", "chr10"), "chr10")
        self.assertEqual(normalise_chrom("chr10", "chr1"), "chr10")
        self.assertEqual(normalise_chrom("10", "1"), "10")


if __name__ == "__main__":
    unittest.main()
