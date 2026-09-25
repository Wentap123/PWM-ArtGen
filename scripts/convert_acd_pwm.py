"""Convert original ACD test annotations to PWM ground truth (no masks required)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from copy import deepcopy
import hashlib
import json

from evaluation.inputs import load_ids
from retrieval.database import CATEGORIES
from retrieval.hashing import HASH_NETWORKX_VERSION
from retrieval.prepare import convert_object, needs_write, read_json, validate_meshes, write_json


CATEGORY_MAPPING = {
    'armoire': 'StorageFurniture', 'bookcase': 'StorageFurniture',
    'chest_of_drawers': 'StorageFurniture', 'desk': 'Table',
    'dishwasher': 'Dishwasher', 'hanging_cabinet': 'StorageFurniture',
    'kitchen_cabinet': 'StorageFurniture', 'microwave': 'Microwave',
    'nightstand': 'StorageFurniture', 'oven': 'Oven',
    'refrigerator': 'Refrigerator', 'sink_cabinet': 'StorageFurniture',
    'tv_stand': 'StorageFurniture', 'washer': 'WashingMachine',
    'table': 'Table', 'cabinet': 'StorageFurniture',
}


def convert_acd_object(original):
    data = deepcopy(original)
    category = data['meta']['obj_cat']
    if category in CATEGORY_MAPPING:
        data['meta']['cat_alias'] = category
        data['meta']['obj_cat'] = CATEGORY_MAPPING[category]
    elif category not in CATEGORIES:
        raise ValueError(f'Unsupported ACD category: {category}')
    return convert_object(data)


def run(args):
    root = args.gt_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    ids = load_ids(args.test_ids, 'acd')
    planned, records = [], []
    for oid in sorted(ids):
        record = {'object_id': oid}
        try:
            source = root / oid / 'object.json'
            if not source.resolve().is_relative_to(root):
                raise ValueError('Object path escapes GT root.')
            data = convert_acd_object(read_json(source))
            validate_meshes(data, source.parent)
            destination = source.with_name('object_pwm.json')
            changed = needs_write(destination, data, args.overwrite)
            if changed:
                planned.append((destination, data))
            record.update(status='valid', parts=len(data['diffuse_tree']), changed=changed,
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          tree_hash=data['meta']['tree_hash'])
        except (ValueError, KeyError, TypeError, OSError) as error:
            record.update(status='failed', error=str(error))
        records.append(record)
    failed = sum(r['status'] == 'failed' for r in records)
    report = dict(dataset='acd', gt_root=str(root), test_ids=str(args.test_ids.resolve()),
                  test_ids_sha256=hashlib.sha256(args.test_ids.read_bytes()).hexdigest(),
                  networkx_version=HASH_NETWORKX_VERSION, expected_objects=len(ids),
                  converted=len(ids)-failed, failed=failed, complete=not failed, objects=records)
    report_path = root / 'pwm_gt_report.json'
    if report_path.is_symlink():
        raise ValueError(f'Unsafe report path: {report_path}')
    print(json.dumps(report, indent=2))
    # Validate all selected objects and overwrite conflicts before writing annotations.
    if not args.dry_run:
        if not failed:
            for path, data in planned:
                write_json(path, data)
        write_json(report_path, report)
    print(f"{'[DRY] ' if args.dry_run else ''}{len(ids)-failed}/{len(ids)} valid GT objects; "
          f"{len(planned) if not failed else 0} planned annotation writes.")
    return int(bool(failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt_root', required=True, type=Path)
    parser.add_argument('--test_ids', type=Path, default=ROOT / 'splits/acd_test.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'[FAIL] {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
