"""Fast benchmark wiring checks without models or full datasets."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.inputs import load_ids, sample_path
from evaluation.benchmark import prepare_job, run
from retrieval.pipeline import write_json
from scripts.prepare_test_ids import collect_ids


class EvaluationTests(unittest.TestCase):
    def test_manifests_and_output_paths(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(len(load_ids(root / 'splits/pm_test.json', 'pm')), 77)
        self.assertEqual(len(load_ids(root / 'splits/acd_test.json', 'acd')), 134)
        self.assertEqual(str(sample_path(dict(oid='12', jid=1, views=['view_idx_03']), 'pm')),
                         '12/joint_1/view_idx_03')
        self.assertEqual(str(sample_path(dict(oid='a', jid=0, data_name='abo-data'), 'acd')),
                         'abo-data/a/joint_0')

    def test_pm_and_acd_assembly(self):
        for dataset in ('pm', 'acd'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                oid = 'Dishwasher/12' if dataset == 'pm' else 'abo-data/armoire/a'
                args = argparse.Namespace(dataset=dataset, gt_root=root/'gt', data_root=root/'data',
                    pred_root=root/'pred', out_dir=root/'out', database_root=root/'db',
                    hashbook=root/'book.json', no_postprocess=False, seed=43,
                    test_ids=root/'ids.json', view_ids=[0], dry_run=True, workers=1)
                (root/'db').mkdir()
                write_json(root/'book.json', {'Dishwasher': {'hash': ['12']}})
                write_json(root/'ids.json', {'dataset': dataset, 'object_ids': [oid]})
                base, door = (0, 1) if dataset == 'pm' else (1, 0)
                tree = [dict(id=base, parent=-1, name='base', children=[door]),
                        dict(id=door, parent=base, name='door', children=[], seg_id=2)]
                write_json(root/'gt'/oid/'object_pwm.json',
                           {'meta': {'obj_cat': 'Dishwasher'}, 'diffuse_tree': tree})
                pred = dict(type=0, axis_dir=[1,0,0], axis_ori=[0,0,0], range=[1],
                            aabb_min=[-1,-1,-1], aabb_max=[1,1,1],
                            aabb_base_min=[-1,-1,-1], aabb_base_max=[1,1,1])
                box = dict(base_bbox_cxcywh_norm=[.5,.5,1,1], joint_bbox_cxcywh_norm=[.5,.5,1,1])
                if dataset == 'pm':
                    write_json(root/'pred/12/joint_1/view_idx_00/geom_pred.json', {'pred': pred})
                    write_json(root/'data/test'/oid/'joint_1/RGB/view_idx_00/info/meta.json', box)
                    self.assertEqual(collect_ids(root/'data', 'pm')['object_ids'], [oid])
                else:
                    write_json(root/'pred/abo-data/a/joint_0/geom_pred.json', {'pred': pred})
                    write_json(root/'data'/oid/'info/10_bbox.json', {'bboxes': {'2': box}})
                job = prepare_job(args, oid, 0 if dataset == 'pm' else None)
                self.assertTrue(job['postprocess'])
                self.assertEqual(len(job['predictions']), 1)
                self.assertEqual(run(args), 0)
                self.assertFalse(args.out_dir.exists())
                args.dry_run = False
                with patch('evaluation.benchmark.process_job', return_value={'status':'ok', 'metrics':{'AOR':None, 'AS-IoU':.5}}):
                    self.assertEqual(run(args), 0)
                report = json.loads((args.out_dir/'metrics.json').read_text())
                self.assertTrue(report['complete'])
                self.assertEqual(report['metrics']['AS-IoU'], .5)


if __name__ == '__main__':
    unittest.main()
