"""Offline single-image regressions: no API, model download, or CUDA required."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import infer_image
from preprocess.parts import bbox, category_of, flatten_tree, sam_parts, sam3_parts
from preprocess.graph_api import parse_response
from preprocess.run import export_inputs, prepare, read_json, write_json


class SingleImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pwm single ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'input image.png'
        Image.new('RGBA', (32, 24), (120, 80, 40, 150)).save(self.image)
        self.output = self.root / 'results'
        self.graph = flatten_tree({'base': [{'drawer': []}, {'door': []}]})
        self.graph['meta'] = {'obj_cat': 'StorageFurniture'}

    def args(self, *extra):
        return infer_image.parser().parse_args(['--image', str(self.image), '--out_dir', str(self.output), *extra])

    def test_api_parsing_and_category(self):
        self.assertEqual(parse_response('<think>reason</think>```json\n{"base": []}\n```'), {'base': []})
        self.assertEqual(category_of({'original_response': 'I recognize all the articulated parts in a washer, they are: base.'}), 'WashingMachine')
        with self.assertRaises(ValueError):
            category_of({}, 'Unknown')
        with self.assertRaises(ValueError):
            parse_response('no JSON')

    def test_invalid_topology(self):
        with self.assertRaises(ValueError):
            flatten_tree({'base': [{'handle': []}]})
        with self.assertRaises(ValueError):
            flatten_tree({'base': [{'door': [{'drawer': []}]}]})

    def test_sam_ids_and_sorting(self):
        labels = np.array([[0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 2, 2]])
        assignment = {'base': {'ids': [0], 'children': [{'drawer': {'ids': [1]}}, {'door': {'ids': [2]}}]}}
        original = deepcopy(self.graph)
        graph, mask = sam_parts(self.graph, assignment, labels)
        self.assertEqual([n['name'] for n in graph['diffuse_tree']], ['base', 'door', 'drawer'])
        np.testing.assert_array_equal(mask, labels + 1)
        self.assertEqual(self.graph, original)
        assignment['base']['ids'] = [1]
        with self.assertRaisesRegex(ValueError, 'more than once'):
            sam_parts(self.graph, assignment, labels)

    def test_sam3_order_and_overlap(self):
        a, b = np.zeros((10, 10)), np.zeros((10, 10))
        a[1:6, 1:6] = 1
        b[4:9, 4:9] = 1
        graph, labels = sam3_parts(self.graph, {'door': [b], 'drawer': [a]})
        self.assertEqual([n['name'] for n in graph['diffuse_tree']], ['base', 'drawer', 'door'])
        self.assertEqual(labels[4, 4], 3)
        with self.assertRaisesRegex(ValueError, 'occluded'):
            sam3_parts(self.graph, {'door': [a], 'drawer': [a]})
        with self.assertRaises(ValueError):
            sam3_parts(self.graph, {})

    def test_sam3_adapter_confidence_fallback(self):
        import torch
        from preprocess.backends import segment_sam3
        calls = []
        class Processor:
            def __init__(self, model, confidence_threshold, device):
                self.threshold = confidence_threshold
            def set_image(self, image):
                return {}
            def reset_all_prompts(self, state):
                pass
            def set_text_prompt(self, state, prompt):
                calls.append((prompt, self.threshold))
                return {'masks': torch.ones(1, 1, 24, 32) if self.threshold == .2 else torch.empty(0, 1, 24, 32)}
        package = types.ModuleType('sam3')
        package.build_sam3_image_model = lambda **kwargs: None
        processor = types.ModuleType('sam3.model.sam3_image_processor')
        processor.Sam3Processor = Processor
        with patch.dict(sys.modules, {'sam3': package, 'sam3.model.sam3_image_processor': processor}):
            masks = segment_sam3(Image.open(self.image), self.graph, 'unused', 'cpu')
        self.assertEqual(calls, [('door', .5), ('door', .2), ('drawer', .5), ('drawer', .2)])
        self.assertEqual(masks['door'].shape, (1, 24, 32))

    def test_sam_grouping_without_model(self):
        from preprocess.sam_regions import get_sam_mask
        labels = np.zeros((32, 32), dtype=bool)
        labels[5:25, 5:25] = True
        class Generator:
            def generate(self, image):
                return [{'area': int(labels.sum()), 'segmentation': labels}]
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., 3] = labels * 255
        groups = get_sam_mask(rgba[..., :3], Generator(), None,
                              rgba_image=Image.fromarray(rgba), size_threshold=10)
        self.assertEqual(set(np.unique(groups)), {-1, 0})
        np.testing.assert_array_equal(groups >= 0, labels)

    def test_bbox_conventions(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:6, 4:10] = True
        np.testing.assert_allclose(bbox(mask, 'sam'), [6.5/20, 3.5/10, 6/20, 4/10])
        np.testing.assert_allclose(bbox(mask, 'sam3'), [7/20, 4/10, 6/20, 4/10])
        with self.assertRaises(ValueError):
            bbox(~np.ones((2, 2), dtype=bool), 'sam')

    def test_export_can_be_loaded_by_pwm(self):
        from datasets.inference import PartNetMobilityInferenceDataset
        a = np.zeros((24, 32))
        a[3:20, 2:14] = 1
        graph, labels = sam3_parts(self.graph, {'door': [a]})
        result = export_inputs(graph, labels, Image.open(self.image), self.output, 'sam3', 'StorageFurniture')
        dataset = PartNetMobilityInferenceDataset(result['data_root'], image_size=(256, 256))
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample['obs']['low_dim'].shape[-1], 8)
        self.assertTrue((self.output / 'data/test/StorageFurniture/input/waiting_use/view_id_00/process/graph_renum.json').is_file())

    def test_defaults_and_interpreter_selection(self):
        output, manifest = infer_image.configure(self.args())
        self.assertEqual(manifest['config']['seg_backend'], 'sam3')
        self.assertFalse(manifest['config']['no_postprocess'])
        with patch.dict(os.environ, {'SAM3_PYTHON': '/sam3/python'}):
            cmds = infer_image.commands(manifest['config'], output, 'all')
        self.assertEqual(cmds[0][1][0], '/sam3/python')
        self.assertNotIn('--no-postprocess', cmds[-1][1])

    def test_dry_run_no_writes(self):
        result = subprocess.run(['bash', str(ROOT / 'demo/infer_image.sh'), '--image', str(self.image),
                                 '--out_dir', str(self.output), '--stage', 'preprocess', '--dry-run',
                                 '--seg_checkpoint', str(self.image)], cwd='/tmp',
                                env=dict(os.environ, PYTHON_BIN=sys.executable), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.output.exists())
        self.assertIn('[preprocess]', result.stdout)

    def test_cached_stages_and_configuration_guard(self):
        output, manifest = infer_image.configure(self.args())
        output.mkdir()
        manifest['stages']['preprocess'] = 'complete'
        write_json(output / 'run.json', manifest)
        write_json(output / 'preprocess/prepared.json', {})
        write_json(output / 'preprocess/data/test/Table/input/waiting_use/view_id_00/joint_1/mask_renum_bbox.json', {})
        with patch.object(infer_image.subprocess, 'run') as runner:
            self.assertEqual(infer_image.run(self.args('--stage', 'preprocess')), 0)
            runner.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'configuration'):
            infer_image.configure(self.args('--seg_backend', 'sam'))
        self.image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'configuration'):
            infer_image.configure(self.args())

    def test_failed_stage_stops_pipeline(self):
        args = self.args('--stage', 'preprocess', '--seg_checkpoint', str(self.image))
        with patch.object(infer_image.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'worker')):
            with self.assertRaises(subprocess.CalledProcessError):
                infer_image.run(args)
        self.assertEqual(read_json(self.output / 'run.json')['stages'], {'preprocess': 'failed'})

    def test_preprocess_cached_api_and_masks(self):
        work = self.root / 'prepared'
        work.mkdir()
        Image.open(self.image).save(work / 'image.png')
        write_json(work / 'graph.json', self.graph)
        masks = np.zeros((1, 24, 32))
        masks[:, 3:20, 3:20] = 1
        np.savez(work / 'sam3_masks.npz', drawer=masks)
        config = dict(seg_backend='sam3', device='cpu', confidence=.5, image=str(self.image),
                      seg_checkpoint='unused', graph_model='unused')
        with patch('preprocess.graph_api.predict_graph', side_effect=AssertionError('API called')), \
             patch('preprocess.backends.segment_sam3', side_effect=AssertionError('model loaded')):
            prepare(config, work)
        self.assertEqual(read_json(work / 'prepared.json')['num_joints'], 1)


if __name__ == '__main__':
    unittest.main()
