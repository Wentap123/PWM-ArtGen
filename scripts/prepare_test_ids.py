"""Freeze benchmark object IDs by listing paths; no image or mesh loading."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json


def collect_ids(root, dataset):
    if dataset == 'pm':
        source = root / 'test'
        ids = [p.relative_to(source).as_posix() for p in source.glob('*/*')
               if p.is_dir() and any(p.glob('joint_*/RGB'))]
    else:
        ids = [p.relative_to(root).as_posix() for group in ('abo-data', 'hssd-data')
               for p in (root / group).glob('*/*') if p.is_dir()
               and (p / 'info/10_bbox.json').is_file()
               and ((p / 'object_pwm.json').is_file() or (p / 'object_uwm.json').is_file())
               and (p / 'rgbs/10.png').is_file() and (p / 'masks/10.png').is_file()]
    if not ids:
        raise ValueError(f'No {dataset} test objects found in {root}')
    return {'dataset': dataset, 'split': 'test',
            'selection': 'rendered test split' if dataset == 'pm' else 'ACD objects with prepared view 10 inputs',
            'object_ids': sorted(set(ids))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['pm', 'acd'], required=True)
    ap.add_argument('--data_root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data = collect_ids(args.data_root, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    print(f"{args.dataset}: {len(data['object_ids'])} objects -> {args.output}")


if __name__ == '__main__':
    main()
