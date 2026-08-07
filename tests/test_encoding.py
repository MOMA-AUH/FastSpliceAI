import unittest

import numpy as np

from spliceai.encoding import one_hot_encode


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
