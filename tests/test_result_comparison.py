from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_results import compare_objects, metric_delta


class ResultComparisonTests(unittest.TestCase):
    def test_metrics_and_missing_aor(self):
        self.assertEqual(metric_delta({'CD': 2.0, 'AOR': None}, {'CD': 1.0, 'AOR': .5}),
                         {'CD': 1.0, 'AOR': None})

    def test_objects_views_and_coverage(self):
        old = {'complete': True, 'objects': [dict(object_id='a', metrics={'CD': 1.0},
                samples=[dict(view=0, metrics={'CD': 1.0}), dict(view=1, metrics={'CD': 1.0})])]}
        new = deepcopy(old)
        new['objects'][0]['metrics']['CD'] = 2.0
        new['objects'][0]['samples'][1]['metrics']['CD'] = 3.0
        result = compare_objects(new, old)['a']
        self.assertEqual(result['metrics_delta_new_minus_old']['CD'], 1.0)
        self.assertEqual(result['per_view_delta_new_minus_old']['view_1']['CD'], 2.0)
        wrong = deepcopy(old)
        wrong['objects'][0]['object_id'] = 'b'
        with self.assertRaises(ValueError):
            compare_objects(new, wrong)
        wrong = deepcopy(old)
        wrong['objects'][0]['samples'].pop()
        with self.assertRaises(ValueError):
            compare_objects(new, wrong)
        wrong = deepcopy(old)
        wrong['complete'] = False
        with self.assertRaises(ValueError):
            compare_objects(new, wrong)


if __name__ == '__main__':
    unittest.main()
