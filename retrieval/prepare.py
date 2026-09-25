"""Convert Singapo object annotations and rebuild the PWM retrieval index."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from .database import ASSETS, CATEGORIES, candidate_ids, load_hashbook
from .hashing import get_hash, HASH_NETWORKX_VERSION


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_tree(nodes):
    if not isinstance(nodes, list) or not nodes:
        raise ValueError('Empty or invalid diffuse_tree.')
    ids = [n['id'] for n in nodes]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Node IDs must be distinct nonnegative integers.')
    by_id = {n['id']: n for n in nodes}
    roots = [n for n in nodes if n.get('parent') == -1]
    if len(roots) != 1 or roots[0].get('name') != 'base':
        raise ValueError('Expected exactly one base root.')
    for node in nodes:
        expected = {n['id'] for n in nodes if n.get('parent') == node['id']}
        children = node.get('children', [])
        if len(children) != len(set(children)) or set(children) != expected:
            raise ValueError('Parent/children references disagree.')
        if node['parent'] != -1 and node['parent'] not in by_id:
            raise ValueError('Missing parent node.')
    seen, pending = set(), [roots[0]['id']]
    while pending:
        index = pending.pop()
        if index in seen:
            raise ValueError('Graph contains a cycle.')
        seen.add(index)
        pending.extend(by_id[index].get('children', []))
    if seen != set(ids):
        raise ValueError('Graph is disconnected or contains a cycle.')
    return roots[0]['id']


def joint_type(node):
    return node.get('type', node.get('joint', {}).get('type', ''))


def convert_object(data):
    """Convert to the merged-handle PWM representation without mutating the source."""
    tree = deepcopy(data['diffuse_tree'])
    root = validate_tree(tree)
    category = data['meta']['obj_cat']
    keep = set()
    for node in tree:
        name, kind = node.get('name', ''), joint_type(node)
        if ((kind == 'fixed' and name == 'base') or
                (name == 'handle' and kind == 'prismatic') or
                (name == 'shelf' and category != 'Dishwasher') or
                (name not in {'shelf', 'knob', 'tray', 'button', 'handle'}
                 and kind not in {'fixed', 'continuous'})):
            keep.add(node['id'])
    by_id = {n['id']: n for n in tree}
    for node in tree:
        if node['id'] not in keep:
            if node['parent'] not in keep:
                raise ValueError(f"Removed node {node['id']} has no retained direct parent; refusing to lose mesh references.")
            parent = by_id[node['parent']]
            for key in ('plys', 'objs'):
                parent[key] = list(dict.fromkeys(parent.get(key, []) + node.get(key, [])))
    retained = [by_id[i] for i in sorted(keep, key=lambda i: (i != root, i))]
    for node in retained:
        if node['parent'] != -1 and node['parent'] not in keep:
            raise ValueError('A retained part lost its parent during conversion.')
        node['children'] = [c for c in node.get('children', []) if c in keep]
        node['name'] = {'handle': 'drawer', 'shelf': 'door'}.get(node['name'], node['name'])
        for key in ('plys', 'objs'):
            if key in node:
                node[key] = list(dict.fromkeys(node[key]))
    mapping = {-1: -1, **{n['id']: i for i, n in enumerate(retained)}}
    for node in retained:
        node['id'], node['parent'] = mapping[node['id']], mapping[node['parent']]
        node['children'] = [mapping[c] for c in node['children']]
    validate_tree(retained)
    meta = deepcopy(data['meta'])
    meta.update(n_diff_parts=len(retained)-1, n_arti_parts=len(retained))
    for kind in ('revolute', 'continuous', 'prismatic', 'screw'):
        meta['n_' + kind] = sum(joint_type(n) == kind for n in retained)
    result = dict(meta=meta, diffuse_tree=retained)
    meta['tree_hash'] = get_hash(result, ignore_handles=False)
    return result


def validate_meshes(data, directory):
    directory = Path(directory).resolve()
    for node in data['diffuse_tree']:
        for key in ('center', 'size'):
            value = node.get('aabb', {}).get(key)
            if not isinstance(value, list) or len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"Invalid AABB for part {node['id']}.")
            # The public PM release contains planar doors with zero thickness.
            # Preserve their geometry; retrieval already clamps scaling divisors.
            if key == 'size' and (any(v < 0 for v in value) or sum(v > 0 for v in value) < 2):
                raise ValueError('AABB sizes must be nonnegative with at least two positive dimensions.')
        if not node.get('plys'):
            raise ValueError(f"Part {node['id']} has no retrieval meshes.")
        for key in ('plys', 'objs'):
            for filename in node.get(key, []):
                path = (directory / filename).resolve()
                if Path(filename).is_absolute() or not path.is_relative_to(directory) or not path.is_file():
                    raise ValueError(f'Missing or unsafe mesh reference: {filename}')


def needs_write(path, value, overwrite):
    if path.is_symlink():
        raise ValueError(f'Refusing to overwrite a symlink: {path}')
    if path.exists():
        try:
            if read_json(path) == value:
                return False
        except (ValueError, OSError):
            pass
        if not overwrite:
            raise FileExistsError(f'Existing output differs: {path}; use --overwrite explicitly.')
    return True


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    # Do not follow pre-existing temporary symlinks.
    if temporary.is_symlink():
        raise ValueError(f'Unsafe temporary path: {temporary}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(args):
    root = args.database_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    reference = load_hashbook(args.reference_hashbook)
    index, converted, details = {}, [], []
    # Only the seven PWM categories; unrelated release categories are left alone.
    for category in CATEGORIES:
        allowed = set(candidate_ids(reference, category))
        for path in sorted((root / category).glob('*/object.json')):
            item = dict(category=category, object_id=path.parent.name)
            try:
                if not path.resolve().is_relative_to(root) or not path.parent.resolve().is_relative_to(root):
                    raise ValueError('Object path escapes the database root.')
                original = read_json(path)
                if original.get('meta', {}).get('obj_cat') != category:
                    raise ValueError('Metadata category does not match the directory.')
                result = convert_object(original)
                validate_meshes(result, path.parent)
                converted.append((path.with_name('object_pwm.json'), result))
                eligible = path.parent.name in allowed
                if eligible:
                    index.setdefault(category, {}).setdefault(result['meta']['tree_hash'], []).append(path.parent.name)
                item.update(status='valid', indexed=eligible, tree_hash=result['meta']['tree_hash'])
            except (ValueError, KeyError, TypeError, OSError) as error:
                item.update(status='failed', error=str(error))
            details.append(item)
    if not details:
        raise ValueError('No <category>/<object_id>/object.json files found in the seven PWM categories.')
    report = dict(networkx_version=HASH_NETWORKX_VERSION,
                  reference_hashbook=str(args.reference_hashbook.resolve()),
                  converted=len(converted), indexed=sum(len(ids) for groups in index.values() for ids in groups.values()),
                  failed=sum(d['status'] == 'failed' for d in details), objects=details)
    # Preflight every overwrite before writing any converted object or new index.
    planned = [(path, value) for path, value in converted + [(root / 'pwm_hash_filtered.json', index)]
               if needs_write(path, value, args.overwrite)]
    report_path = root / 'pwm_database_report.json'
    if report_path.is_symlink():
        raise ValueError(f'Unsafe report path: {report_path}')
    print(json.dumps(report, indent=2))
    print(f"{'[DRY] ' if args.dry_run else ''}{len(planned)} changed outputs; {report['indexed']} indexed objects.")
    if not args.dry_run:
        for path, value in planned:
            write_json(path, value)
        write_json(report_path, report)
    return int(report['failed'] > 0 or report['indexed'] == 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database_root', required=True, type=Path)
    parser.add_argument('--reference_hashbook', type=Path, default=ASSETS / 'singapo_retrieval_reference.json',
                        help='Original category/hash/IDs index; only category and object membership are used.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'[FAIL] {error}')
        return 1
