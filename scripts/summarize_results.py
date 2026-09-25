"""Audit full benchmark coverage and save reproducibility metadata and metrics."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
from retrieval.evaluate import aggregate_metrics


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def metric_delta(current, previous):
    return {key: (value - previous[key] if value is not None and previous[key] is not None else None)
            for key, value in current.items()}


def compare_objects(current, previous):
    old = {item['object_id']: item for item in previous['objects']}
    if not previous['complete'] or set(old) != {item['object_id'] for item in current['objects']}:
        raise ValueError('Cannot compare incomplete or different object lists')
    comparison = {}
    for item in current['objects']:
        baseline = old[item['object_id']]
        views = {sample['view']: sample for sample in baseline['samples']}
        if set(views) != {sample['view'] for sample in item['samples']}:
            raise ValueError('Cannot compare different evaluation views')
        comparison[item['object_id']] = {
            'metrics_delta_new_minus_old': metric_delta(item['metrics'], baseline['metrics']),
            'per_view_delta_new_minus_old': {
                f"view_{sample['view']}": metric_delta(sample['metrics'], views[sample['view']]['metrics'])
                for sample in item['samples']}}
    return comparison


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out_root', required=True, type=Path)
    ap.add_argument('--datasets_root', required=True, type=Path)
    ap.add_argument('--ckpt', type=Path, default=ROOT/'checkpoints/pwm_final.pt')
    ap.add_argument('--hashbook', type=Path, default=ROOT/'retrieval/assets/pwm_hash_filtered.json')
    ap.add_argument('--compare_summary', type=Path,
                    help='Previous summary; adjacent dataset/evaluation/metrics.json supplies object comparisons.')
    args = ap.parse_args()
    summary = {'checkpoint':str(args.ckpt.resolve()), 'checkpoint_sha256':digest(args.ckpt),
               'retrieval_hashbook':str(args.hashbook.resolve()), 'retrieval_hashbook_sha256':digest(args.hashbook),
               'postprocess':True, 'views':[0,1], 'sample_steps':10, 'inference_seed':43, 'datasets':{}}
    summary['metric_note'] = 'The local Singapo AS-IoU and RS-IoU implementation returns 1 - GIoU (lower is better), not ordinary IoU. Distances and AOR are also lower-is-better.'
    for dataset in ('pm', 'acd'):
        data = args.datasets_root/dataset
        ids = read(data/'test_ids.json')['object_ids']
        report = read(args.out_root/dataset/'evaluation/metrics.json')
        if not report['complete'] or report['failed'] or report['succeeded'] != len(ids):
            raise ValueError(f'{dataset}: incomplete evaluation')
        if {r['object_id'] for r in report['objects']} != set(ids):
            raise ValueError(f'{dataset}: metric/test-ID mismatch')
        count = 0
        for oid in ids:
            views = read(data/oid/'views.json')['views']
            if set(views) != {'view_0','view_1'}:
                raise ValueError(f'{oid}: expected exactly two views')
            for view, info in views.items():
                for sample in info['samples']:
                    path = args.out_root/dataset/'predictions'/oid/view/f"joint_{sample['joint_id']}"/'geom_pred.json'
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    count += 1
        for item in report['objects']:
            if {s['view'] for s in item['samples']} != {0,1}:
                raise ValueError(f'{dataset}: missing view metrics')
        per_view = {f'view_{v}':aggregate_metrics([s['metrics'] for item in report['objects']
                    for s in item['samples'] if s['view']==v]) for v in (0,1)}
        summary['datasets'][dataset] = dict(objects=len(ids), joint_view_predictions=count,
            data_selection=read(data/'dataset.json')['selection'],
            test_ids_sha256=digest(data/'test_ids.json'), data_checksums_sha256=digest(data/'checksums.json'),
            evaluation_seed=report['seed'], metrics=report['metrics'], per_view_metrics=per_view)
    if args.compare_summary:
        previous = read(args.compare_summary)
        summary['comparison'] = {'previous_summary':str(args.compare_summary.resolve()),
                                 'previous_summary_sha256':digest(args.compare_summary), 'datasets':{}}
        for name, current in summary['datasets'].items():
            old = previous['datasets'][name]
            if old['test_ids_sha256'] != current['test_ids_sha256']:
                raise ValueError(f'{name}: cannot directly compare different test ID lists')
            if old['data_checksums_sha256'] != current['data_checksums_sha256']:
                raise ValueError(f'{name}: cannot directly compare different input data')
            old_report = read(args.compare_summary.parent/name/'evaluation/metrics.json')
            new_report = read(args.out_root/name/'evaluation/metrics.json')
            summary['comparison']['datasets'][name] = {
                'metrics_delta_new_minus_old': metric_delta(current['metrics'], old['metrics']),
                'objects': compare_objects(new_report, old_report),
                'per_view_delta_new_minus_old': {
                    view: {key: value - old['per_view_metrics'][view][key]
                           for key, value in metrics.items() if value is not None and old['per_view_metrics'][view][key] is not None}
                    for view, metrics in current['per_view_metrics'].items()}}
    output = args.out_root/'summary.json'
    with output.open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
