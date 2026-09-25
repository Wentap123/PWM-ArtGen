"""Ground-truth graph benchmark protocol; not the photo/SAM inference protocol."""
import argparse
from copy import deepcopy
from pathlib import Path

from evaluation.inputs import ROOT, load_ids
from retrieval.assemble import assemble_object
from retrieval.database import object_json_path, resolve_hashbook, load_hashbook
from retrieval.evaluate import aggregate_metrics
from retrieval.pipeline import read_json, write_json, validate_prediction, process_job


def prepare_job(args, object_id, view):
    if (args.data_root / 'dataset.json').is_file():
        return prepare_portable_job(args, object_id, view)
    gt_dir = args.gt_root / object_id
    graph = deepcopy(read_json(object_json_path(gt_dir)))
    # Strip GT geometry in assemble_object; only topology and names are used.
    tree = sorted(graph['diffuse_tree'], key=lambda node: node['id'])
    graph['diffuse_tree'] = tree
    nodes = [node for node in tree if node['name'] != 'base']
    category = graph['meta']['obj_cat']
    parts = object_id.split('/')
    oid = parts[-1]
    pred_dir = args.pred_root / oid if args.dataset == 'pm' else args.pred_root / parts[0] / oid
    joints = sorted((p for p in pred_dir.glob('joint_*') if p.is_dir()),
                    key=lambda p: int(p.name.removeprefix('joint_')))
    if len(joints) != len(nodes):
        raise ValueError(f'{object_id}: expected {len(nodes)} joints, got {len(joints)}')
    predictions, boxes = [], []
    acd_boxes = read_json(args.data_root / object_id / 'info/10_bbox.json')['bboxes'] if args.dataset == 'acd' else None
    for node, joint in zip(nodes, joints):
        jid = int(joint.name.removeprefix('joint_'))
        if args.dataset == 'acd' and jid != node['id']:
            raise ValueError(f'ACD joint/node mismatch: {joint} vs {node["id"]}')
        source = joint / f'view_idx_{view:02d}' if view is not None else joint
        pred = read_json(source / 'geom_pred.json')['pred']
        validate_prediction(pred)
        predictions.append(pred)
        if not args.no_postprocess:
            box = (read_json(args.data_root / 'test' / object_id / joint.name / 'RGB' /
                             f'view_idx_{view:02d}' / 'info/meta.json') if view is not None
                   else acd_boxes[str(node['seg_id'])])
            for key in ('base_bbox_cxcywh_norm', 'joint_bbox_cxcywh_norm'):
                if not isinstance(box.get(key), list) or len(box[key]) != 4:
                    raise ValueError(f'Invalid {key}: {object_id}/{joint.name}')
            boxes.append(box)
    assemble_object(graph, predictions)
    output = args.out_dir / ('0@' + object_id.replace('/', '@')) / str(view if view is not None else 0)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    return dict(object_id=f'{object_id}@{view}', category=category, graph=graph,
                predictions=predictions, boxes=boxes, keys=[j.name for j in joints],
                output=str(output), postprocess=not args.no_postprocess,
                database_root=str(args.database_root), evaluate=True, gt_dir=str(gt_dir),
                seed=args.seed, hashbook=str(args.hashbook))


def prepare_portable_job(args, object_id, view):
    source = args.data_root / object_id
    graph = read_json(source / 'graph.json')
    records = read_json(source / 'views.json')['views'][f'view_{view}']['samples']
    records = sorted(records, key=lambda r:r['node_id'])
    graph['diffuse_tree'].sort(key=lambda n:n['id'])
    nodes = [n['id'] for n in graph['diffuse_tree'] if n['name'] != 'base']
    if [r['node_id'] for r in records] != nodes:
        raise ValueError(f'Graph/joint mapping mismatch: {object_id}')
    predictions = []
    for record in records:
        path = args.pred_root / object_id / f'view_{view}' / f"joint_{record['joint_id']}" / 'geom_pred.json'
        pred = read_json(path)['pred']
        validate_prediction(pred)
        predictions.append(pred)
    assemble_object(graph, predictions)
    gt_dir = args.gt_root / object_id
    if not object_json_path(gt_dir).is_file():
        raise FileNotFoundError(gt_dir)
    output = args.out_dir / ('0@' + object_id.replace('/', '@')) / str(view)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    return dict(object_id=f'{object_id}@{view}', category=graph['meta']['obj_cat'], graph=graph,
                predictions=predictions, boxes=records, keys=[str(r['joint_id']) for r in records],
                output=str(output), postprocess=not args.no_postprocess, database_root=str(args.database_root),
                evaluate=True, gt_dir=str(gt_dir), seed=args.seed, hashbook=str(args.hashbook))


def evaluate_one(item):
    args, oid = item
    cached = getattr(args, 'completed', {}).get(oid)
    if cached is not None:
        for sample in cached['samples']:
            directory = Path(sample['output'])
            if not (directory / 'object.ply').is_file() or not (directory / 'object_pwm.json').is_file():
                raise ValueError(f'Incomplete cached meshes: {directory}')
            if read_json(directory / 'metrics.json') != sample['metrics']:
                raise ValueError(f'Cached metrics changed: {directory}')
        print(f'[cached] {oid}', flush=True)
        return cached
    record = {'object_id': oid, 'samples': [], 'status': 'ok'}
    for view in (args.view_ids if args.dataset == 'pm' or (args.data_root / 'dataset.json').is_file() else [None]):
        try:
            job = prepare_job(args, oid, view)
            result = {'status': 'validated'} if args.dry_run else process_job(job)
            record['samples'].append(dict(view=view, **result))
        except Exception as error:
            record['status'] = 'failed'
            record['samples'].append(dict(view=view, status='failed', error=str(error)))
            print(f'[FAIL] {oid}, view={view}: {error}', flush=True)
    if record['status'] == 'ok' and not args.dry_run:
        record['metrics'] = aggregate_metrics([s['metrics'] for s in record['samples']])
    print(f"[{record['status']}] {oid}", flush=True)
    return record


def run(args):
    for key in ('data_root', 'gt_root', 'pred_root', 'database_root', 'out_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for source in (args.data_root, args.gt_root, args.pred_root, args.database_root):
        if not source.is_dir():
            raise FileNotFoundError(source)
        if args.out_dir.is_relative_to(source) or source.is_relative_to(args.out_dir):
            raise ValueError('Output must be separate from data, GT, predictions and database.')
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not getattr(args, 'resume', False):
        raise FileExistsError('Use a fresh --out_dir to avoid mixing evaluations.')
    args.hashbook = resolve_hashbook(args.database_root, args.hashbook)
    load_hashbook(args.hashbook)
    default_ids = args.data_root / 'test_ids.json' if (args.data_root / 'dataset.json').is_file() else ROOT / 'splits' / f'{args.dataset}_test.json'
    ids = load_ids(args.test_ids or default_ids, args.dataset)
    if getattr(args, 'resume', False):
        previous = read_json(args.out_dir / 'metrics.json')
        if previous['dataset'] != args.dataset or previous['seed'] != args.seed or {r['object_id'] for r in previous['objects']} != ids:
            raise ValueError('Resume requires identical dataset, test IDs and seed.')
        args.completed = {r['object_id']:r for r in previous['objects'] if r['status']=='ok'}
        wanted = set(args.view_ids) if (args.data_root / 'dataset.json').is_file() or args.dataset=='pm' else {None}
        if any({s['view'] for s in r['samples']} != wanted for r in args.completed.values()):
            raise ValueError('Resume requires identical views.')
        if not args.dry_run:
            backup = args.out_dir / 'metrics.before_resume.json'
            if backup.exists():
                raise FileExistsError(f'Previous resume backup already exists: {backup}')
            write_json(backup, previous)
    items = [(args, oid) for oid in sorted(ids)]
    if args.workers == 1:
        results = [evaluate_one(item) for item in items]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(evaluate_one, items))
    good = [r for r in results if r['status'] == 'ok']
    if not args.dry_run:
        write_json(args.out_dir / 'metrics.json', {
            'dataset': args.dataset, 'seed': args.seed, 'expected_objects': len(ids),
            'succeeded': len(good), 'failed': len(ids) - len(good),
            'complete': len(good) == len(ids),
            'aggregation': 'mean over views per object, then mean over successful objects',
            'metrics': aggregate_metrics([r['metrics'] for r in good]) if good else None,
            'objects': results})
    return int(len(good) != len(ids))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=('pm', 'acd'))
    for name in ('data_root', 'gt_root', 'pred_root', 'database_root', 'out_dir'):
        ap.add_argument('--' + name, required=True, type=Path)
    ap.add_argument('--test_ids', type=Path)
    ap.add_argument('--view_ids', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--hashbook', type=Path)
    ap.add_argument('--no-postprocess', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume', action='store_true', help='Retry failed objects; requires unchanged input data, predictions, database and settings.')
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error('--workers must be positive')
    try:
        return run(args)
    except (ValueError, OSError) as error:
        ap.exit(1, f'Error: {error}\n')
