"""Frontal view selection uses only masks, not predictions or metric results."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.frontal import rank_views, normalized_labels, similarity


class FrontalTests(unittest.TestCase):
    def test_layout_and_aspect_preserving_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'masks').mkdir()
            reference = np.zeros((100, 100), dtype=np.uint8)
            reference[20:80,20:50] = 1
            reference[20:80,50:80] = 2
            side = np.zeros_like(reference)
            side[20:80,40:50] = 1
            side[20:80,50:60] = 2
            for name, mask in [('10',reference),('03',reference),('07',reference),('00',side)]:
                Image.fromarray(mask).save(root/'masks'/f'{name}.png')
            ranked = rank_views(root, ['07','00','03'], [1,2])
            self.assertEqual([x[1] for x in ranked], ['03','07','00'])
            self.assertEqual(ranked[0][0], 1)
            normalized = normalized_labels(root/'masks/10.png')
            self.assertEqual(similarity(normalized, normalized, [1,2]), 1)


if __name__ == '__main__':
    unittest.main()
