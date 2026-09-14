import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codec import decode, pack_trits, quantize, unpack_trits


class CodecTests(unittest.TestCase):
    def test_all_codewords_against_scalar_definition(self):
        words = np.array(list(itertools.product([-1, 0, 1], repeat=5)), dtype=np.int8)
        expected = [sum((int(t) + 1) * 3**i for i, t in enumerate(row)) for row in words]
        packed = pack_trits(words)
        np.testing.assert_array_equal(packed[:, 0], expected)
        self.assertEqual(len(set(expected)), 243)
        np.testing.assert_array_equal(unpack_trits(packed, 5), words)

    def test_tails_and_group_boundaries(self):
        rng = np.random.default_rng(123)
        for size in [1, 2, 3, 4, 5, 6, 32, 64, 128]:
            trits = rng.integers(-1, 2, (11, size), dtype=np.int8)
            np.testing.assert_array_equal(unpack_trits(pack_trits(trits), size), trits)

    def test_invalid_bytes_and_padding(self):
        for value in range(243, 256):
            with self.assertRaises(ValueError):
                unpack_trits(np.array([[value]], dtype=np.uint8), 5)
        with self.assertRaises(ValueError):
            unpack_trits(np.array([[0]], dtype=np.uint8), 4)
        with self.assertRaises(ValueError):
            unpack_trits(np.array([[121]], dtype=np.uint8), 6)

    def test_written_bytes_and_scale_rounding(self):
        weight = np.array([[-0.3, 0, 0.1, 0.2, 0.7, -0.6]], dtype=np.float32)
        trits, scales = quantize(weight, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "group.npz"
            np.savez(path, codes=pack_trits(trits), scales=scales)
            with np.load(path, allow_pickle=False) as package:
                decoded = decode(package["codes"], package["scales"], weight.shape, 3)
                np.testing.assert_array_equal(decoded, (trits * scales.astype(np.float32)[:, None]).reshape(weight.shape))
                # A valid changed code byte must change the reconstruction.
                changed = package["codes"].copy()
                changed[0, 0] += 1
                self.assertFalse(np.array_equal(decode(changed, package["scales"], weight.shape, 3), decoded))

    def test_zero_groups_and_bad_inputs(self):
        trits, scales = quantize(np.zeros((2, 64)), 64)
        self.assertTrue(np.all(trits == 0))
        self.assertTrue(np.all(decode(pack_trits(trits), scales, (2, 64), 64) == 0))
        for value in [np.nan, np.inf]:
            with self.assertRaises(ValueError):
                quantize(np.full((1, 64), value), 64)


if __name__ == "__main__":
    unittest.main()
