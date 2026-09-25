"""Raw ACD GT conversion, independent of segmentation and historical annotations."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.convert_acd_pwm import CATEGORY_MAPPING, convert_acd_object, main
from retrieval.hashing import HASH_NETWORKX_VERSION


@unittest.skipUnless(nx.__version__ == HASH_NETWORKX_VERSION, 'Requires retrieval environment.')
class ACDConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.oid = 'abo-data/armoire/example'
        self.directory = self.root / self.oid
        self.directory.mkdir(parents=True)
        self.data = {'meta': {'obj_cat': 'armoire'}, 'diffuse_tree': [
            self.node(0, 'door', 'revolute', 2, [1]),
            self.node(1, 'handle', 'fixed', 0, []),
            self.node(2, 'base', 'fixed', -1, [0])]}
        self.source = self.directory / 'object.json'
        self.source.write_text(json.dumps(self.data))
        for node in self.data['diffuse_tree']:
            (self.directory / node['plys'][0]).write_text('synthetic mesh reference')
        self.ids = self.root / 'ids.json'
        self.ids.write_text(json.dumps({'dataset': 'acd', 'object_ids': [self.oid]}))

    @staticmethod
    def node(index, name, kind, parent, children):
        return dict(id=index, name=name, parent=parent, children=children,
                    joint={'type': kind, 'range': [0, 90]},
                    aabb={'center': [0, 0, 0], 'size': [1, 1, 1]}, plys=[f'{index}.ply'])

    def run_conversion(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(['--gt_root', str(self.root), '--test_ids', str(self.ids), *extra])

    def test_mapping_merge_and_geometry(self):
        original = deepcopy(self.data)
        result = convert_acd_object(self.data)
        self.assertEqual(self.data, original)
        self.assertEqual(result['meta']['obj_cat'], 'StorageFurniture')
        self.assertEqual(result['meta']['cat_alias'], 'armoire')
        self.assertEqual([n['name'] for n in result['diffuse_tree']], ['base', 'door'])
        door = result['diffuse_tree'][1]
        self.assertEqual(door['plys'], ['0.ply', '1.ply'])
        self.assertEqual(door['joint'], original['diffuse_tree'][0]['joint'])
        self.assertEqual(door['aabb'], original['diffuse_tree'][0]['aabb'])
        for category, target in CATEGORY_MAPPING.items():
            data = deepcopy(self.data)
            data['meta']['obj_cat'] = category
            self.assertEqual(convert_acd_object(data)['meta']['obj_cat'], target)

    def test_raw_only_idempotence_and_pwm_name(self):
        before = self.source.read_bytes()
        self.assertEqual(self.run_conversion(), 0)
        output = self.directory / 'object_pwm.json'
        mtime = output.stat().st_mtime_ns
        self.assertEqual(self.run_conversion(), 0)
        self.assertEqual(output.stat().st_mtime_ns, mtime)
        self.assertEqual(before, self.source.read_bytes())
        self.assertFalse((self.directory / 'object_uwm.json').exists())
        self.assertTrue(json.loads((self.root / 'pwm_gt_report.json').read_text())['complete'])

    def test_dry_run_and_external_directory(self):
        before = sorted(str(p) for p in self.root.rglob('*'))
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/convert_acd_pwm.py'),
                                 '--gt_root', str(self.root), '--test_ids', str(self.ids), '--dry-run'],
                                cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))

    def test_missing_mesh_or_object_prevents_annotation_writes(self):
        (self.directory / '1.ply').unlink()
        self.assertEqual(self.run_conversion(), 1)
        self.assertFalse((self.directory / 'object_pwm.json').exists())
        (self.directory / '1.ply').write_text('mesh')
        self.ids.write_text(json.dumps({'dataset': 'acd', 'object_ids': [self.oid, 'hssd-data/desk/missing']}))
        self.assertEqual(self.run_conversion(), 1)
        self.assertFalse((self.directory / 'object_pwm.json').exists())

    def test_conflict_and_unknown_category(self):
        output = self.directory / 'object_pwm.json'
        output.write_text('{}')
        self.assertEqual(self.run_conversion(), 1)
        self.assertEqual(output.read_text(), '{}')
        self.assertEqual(self.run_conversion('--overwrite'), 0)
        self.data['meta']['obj_cat'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            convert_acd_object(self.data)


if __name__ == '__main__':
    unittest.main()
