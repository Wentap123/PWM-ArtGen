"""Synthetic database tests; never read or modify the real Singapo database."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval.database import candidate_ids, load_hashbook, resolve_hashbook
from retrieval.hashing import get_hash, HASH_NETWORKX_VERSION
from retrieval.prepare import convert_object, main, validate_meshes
from demo import infer_image


def node(i, name, kind, parent=-1, children=()):
    return dict(id=i, name=name, parent=parent, children=list(children),
                joint={'type': kind, 'axis': {'origin': [0, 0, 0], 'direction': [0, 0, 1]}, 'range': [0, 90]},
                aabb={'center': [0, 0, 0], 'size': [1, 1, 1]},
                plys=[f'plys/{i}.ply'], objs=[f'objs/{i}.obj'])


def source(category='Table'):
    return {'meta': {'obj_cat': category, 'tree_hash': 'OLD'}, 'diffuse_tree': [
        node(10, 'base', 'fixed', children=[20, 40]),
        node(20, 'door', 'revolute', 10, [30]),
        node(30, 'handle', 'fixed', 20),
        node(40, 'handle', 'prismatic', 10)]}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@unittest.skipUnless(nx.__version__ == HASH_NETWORKX_VERSION,
                     'Run database tests in the singapo environment (networkx==3.4.2).')
class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pwm database ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / 'database'
        self.database.mkdir()
        self.reference = self.root / 'reference.json'
        write(self.reference, {'Table': {'OLD': ['123']}})
        self.directory = self.database / 'Table/123'
        self.data = source()
        self.add_object(self.directory, self.data)

    def add_object(self, directory, data):
        write(directory / 'object.json', data)
        for part in data['diffuse_tree']:
            for key in ('plys', 'objs'):
                for filename in part[key]:
                    path = directory / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('fixture mesh reference\n')

    def args(self, *extra):
        return ['--database_root', str(self.database), '--reference_hashbook', str(self.reference), *extra]

    def run_prepare(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(self.args(*extra))

    def test_merge_and_geometry_preservation(self):
        original = deepcopy(self.data)
        result = convert_object(self.data)
        nodes = result['diffuse_tree']
        self.assertEqual([n['name'] for n in nodes], ['base', 'door', 'drawer'])
        self.assertEqual(nodes[1]['plys'], ['plys/20.ply', 'plys/30.ply'])
        self.assertEqual(nodes[1]['objs'], ['objs/20.obj', 'objs/30.obj'])
        self.assertEqual(nodes[1]['aabb'], self.data['diffuse_tree'][1]['aabb'])
        self.assertEqual(nodes[1]['joint'], self.data['diffuse_tree'][1]['joint'])
        self.assertEqual(nodes[0]['children'], [1, 2])
        self.assertEqual(result['meta']['n_prismatic'], 1)
        self.assertEqual(result['meta']['n_diff_parts'], 2)
        self.assertEqual(self.data, original)

    def test_hash_is_rebuilt_after_removal(self):
        result = convert_object(self.data)
        self.assertEqual(result['meta']['tree_hash'], get_hash(result))
        self.assertNotEqual(result['meta']['tree_hash'], get_hash(self.data, ignore_handles=False))

    def test_shelf_exception_and_deduplication(self):
        self.data['diffuse_tree'][3]['name'] = 'shelf'
        self.assertEqual(convert_object(self.data)['diffuse_tree'][2]['name'], 'door')
        self.data['meta']['obj_cat'] = 'Dishwasher'
        result = convert_object(self.data)
        self.assertEqual(len(result['diffuse_tree']), 2)
        self.assertIn('plys/40.ply', result['diffuse_tree'][0]['plys'])
        self.data['diffuse_tree'][1]['plys'].append('plys/30.ply')
        self.assertEqual(convert_object(self.data)['diffuse_tree'][1]['plys'].count('plys/30.ply'), 1)

    def test_invalid_topology_and_removed_parent(self):
        invalid = deepcopy(self.data)
        invalid['diffuse_tree'][3]['id'] = 20
        with self.assertRaises(ValueError):
            convert_object(invalid)
        invalid = deepcopy(self.data)
        invalid['diffuse_tree'][1]['joint']['type'] = 'fixed'
        with self.assertRaisesRegex(ValueError, 'retained direct parent'):
            convert_object(invalid)
        invalid = deepcopy(self.data)
        invalid['diffuse_tree'][3]['parent'] = -1
        invalid['diffuse_tree'][0]['children'] = [20]
        with self.assertRaisesRegex(ValueError, 'one base root'):
            convert_object(invalid)
        cycle = {'meta': {'obj_cat': 'Table'}, 'diffuse_tree': [
            node(0, 'base', 'fixed'), node(1, 'door', 'revolute', 2, [2]),
            node(2, 'drawer', 'prismatic', 1, [1])]}
        with self.assertRaisesRegex(ValueError, 'cycle'):
            convert_object(cycle)

    def test_missing_and_unsafe_meshes(self):
        value = convert_object(self.data)
        (self.directory / 'plys/30.ply').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            validate_meshes(value, self.directory)
        value['diffuse_tree'][1]['plys'] = ['../../escape.ply']
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            validate_meshes(value, self.directory)

    def test_dry_run_writes_nothing(self):
        before = {str(p): p.read_bytes() for p in self.database.rglob('*') if p.is_file()}
        self.assertEqual(self.run_prepare('--dry-run'), 0)
        after = {str(p): p.read_bytes() for p in self.database.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_planar_source_boxes_preserved(self):
        self.data['diffuse_tree'][1]['aabb']['size'] = [1, 1, 0]
        write(self.directory / 'object.json', self.data)
        original = (self.directory / 'object.json').read_bytes()
        self.assertEqual(self.run_prepare(), 0)
        result = json.loads((self.directory / 'object_pwm.json').read_text())
        self.assertEqual(result['diffuse_tree'][1]['aabb']['size'], [1, 1, 0])
        self.assertEqual((self.directory / 'object.json').read_bytes(), original)
        for invalid in ([1, 1, -1], [1, 0, 0], [0, 0, 0]):
            result['diffuse_tree'][1]['aabb']['size'] = invalid
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                validate_meshes(result, self.directory)

    def test_conversion_index_membership_and_idempotence(self):
        self.add_object(self.database / 'Table/999', source())
        original = (self.directory / 'object.json').read_bytes()
        self.assertEqual(self.run_prepare(), 0)
        book = load_hashbook(self.database / 'pwm_hash_filtered.json')
        self.assertEqual(candidate_ids(book, 'Table'), ['123'])
        converted = json.loads((self.directory / 'object_pwm.json').read_text())
        self.assertEqual(book, {'Table': {converted['meta']['tree_hash']: ['123']}})
        self.assertTrue((self.database / 'Table/999/object_pwm.json').is_file())
        paths = [self.directory / 'object_pwm.json', self.database / 'pwm_hash_filtered.json']
        times = [p.stat().st_mtime_ns for p in paths]
        self.assertEqual(self.run_prepare(), 0)
        self.assertEqual(times, [p.stat().st_mtime_ns for p in paths])
        self.assertEqual(original, (self.directory / 'object.json').read_bytes())

    def test_conflict_preflight_and_explicit_overwrite(self):
        existing = self.directory / 'object_pwm.json'
        write(existing, {'unrelated': True})
        self.assertEqual(self.run_prepare(), 1)
        self.assertFalse((self.database / 'pwm_hash_filtered.json').exists())
        self.assertEqual(json.loads(existing.read_text()), {'unrelated': True})
        self.assertEqual(self.run_prepare('--overwrite'), 0)
        write(self.database / 'pwm_hash_filtered.json', {'Table': {'obsolete': ['123']}})
        before = existing.read_bytes()
        self.assertEqual(self.run_prepare(), 1)
        self.assertEqual(existing.read_bytes(), before)

    def test_failed_object_excluded(self):
        self.add_object(self.database / 'Table/broken', source())
        (self.database / 'Table/broken/plys/30.ply').unlink()
        write(self.reference, {'Table': {'OLD': ['123', 'broken']}})
        self.assertEqual(self.run_prepare(), 1)
        self.assertEqual(candidate_ids(load_hashbook(self.database / 'pwm_hash_filtered.json'), 'Table'), ['123'])
        report = json.loads((self.database / 'pwm_database_report.json').read_text())
        self.assertEqual(report['failed'], 1)

    def test_index_selection_and_unsafe_ids(self):
        self.assertEqual(resolve_hashbook(self.database).parent.name, 'assets')
        local = self.database / 'pwm_hash_filtered.json'
        write(local, {})
        self.assertEqual(resolve_hashbook(self.database), local)
        self.assertEqual(resolve_hashbook(self.database, self.reference), self.reference)
        with self.assertRaises(FileNotFoundError):
            resolve_hashbook(self.database, self.root / 'missing.json')
        write(local, {'Table': {'hash': ['../bad']}})
        with self.assertRaises(ValueError):
            load_hashbook(local)

    def test_entrypoint_from_another_directory(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_database.py'), *self.args('--dry-run')],
                                cwd='/tmp', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.directory / 'object_pwm.json').exists())

    def test_single_image_forwards_custom_index(self):
        config = dict(seg_backend='sam3', ckpt='unused', no_postprocess=False, hashbook=str(self.reference))
        command = infer_image.commands(config, self.root, 'retrieval')[0][1]
        self.assertEqual(command[-2:], ['--hashbook', str(self.reference)])

    def test_generated_database_retrieves_real_meshes(self):
        import trimesh
        from retrieval.mesh import retrieve_meshes
        # All data and meshes are synthetic; exercise real metrics and mesh export.
        for part in self.data['diffuse_tree']:
            trimesh.creation.box().export(self.directory / part['plys'][0])
        self.assertEqual(self.run_prepare(), 0)
        requirement = json.loads((self.directory / 'object_pwm.json').read_text())
        output = self.root / 'retrieved'
        with contextlib.redirect_stdout(io.StringIO()):
            result = retrieve_meshes(requirement, self.database, output, resolve_hashbook(self.database))
        self.assertEqual(len(result['diffuse_tree']), 3)
        self.assertTrue((output / 'object.ply').is_file())
        for part in result['diffuse_tree']:
            self.assertTrue((output / part['plys'][0]).is_file())
            self.assertGreater(len(trimesh.load(output / part['plys'][0], force='mesh').vertices), 0)

    def test_both_retrieval_stages_respect_index(self):
        from retrieval.singapo import obj_retrieval as module
        metric = Mock(return_value={k: 0.0 for k in ('AS-cDist', 'AS-IoU', 'RS-IoU', 'RS-cDist')})
        converted = convert_object(self.data)
        write(self.directory / 'object_pwm.json', converted)
        excluded = self.database / 'Table/999'
        self.add_object(excluded, source())
        write(excluded / 'object_pwm.json', converted)
        # No matching hash: search must still stay within the allowed IDs.
        write(self.reference, {'Table': {'different': ['123']}})
        with contextlib.redirect_stdout(io.StringIO()), patch.object(module, 'IoU_cDist', metric):
            candidates = module.find_obj_candidates(converted, str(self.database), self.reference)
            parts = module.pick_and_rescale_parts(converted, [], str(self.database), hashbook_path=self.reference, verbose=False)
        self.assertEqual([Path(c['dir']).name for c in candidates], ['123'])
        self.assertTrue(all(Path(p['dir']).name == '123' for p in parts))
        self.assertEqual(metric.call_count, 1)

    def test_cross_category_fallback_when_first_category_has_no_matching_part(self):
        from retrieval.singapo import obj_retrieval as module
        converted = convert_object(self.data)
        write(self.directory / 'object_pwm.json', converted)
        write(self.reference, {'Oven': {}, 'Table': {'any': ['123']}})
        requirement = deepcopy(converted)
        requirement['meta']['obj_cat'] = 'Oven'
        (self.database / 'Oven').mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            parts = module.pick_and_rescale_parts(requirement, [], str(self.database),
                                                  hashbook_path=self.reference, verbose=False)
        self.assertTrue(all(Path(p['dir']).name == '123' for p in parts))


if __name__ == '__main__':
    unittest.main()
