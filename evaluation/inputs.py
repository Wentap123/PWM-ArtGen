"""Benchmark-specific loading, shared by the normal PWM inference engine."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_ids(path, dataset):
    with Path(path).open() as stream:
        manifest = json.load(stream)
    if manifest.get('dataset') != dataset:
        raise ValueError(f'Test ID manifest must have dataset={dataset!r}.')
    ids = manifest['object_ids']
    depth = 2 if dataset == 'pm' else 3
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Test IDs must be nonempty and unique.')
    for item in ids:
        if len(item.split('/')) != depth or any(p in ('', '.', '..') for p in item.split('/')) or '\\' in item:
            raise ValueError(f'Invalid test ID: {item}')
    return set(ids)


def build_dataset(args):
    from datasets.partnet_mobility_action import PartNetMobilityActionsDataset
    from datasets.acd import ACDInferenceDataset
    root = Path(args.data_root)
    default_ids = root / 'test_ids.json' if (root / 'dataset.json').is_file() else ROOT / 'splits' / f'{args.dataset}_test.json'
    ids = load_ids(args.test_ids or default_ids, args.dataset)
    if (root / 'dataset.json').is_file():
        from datasets.benchmark import BenchmarkDataset
        return BenchmarkDataset(root, args.dataset, ids, args.view_ids, args.img_size)
    source = root / args.split if args.dataset == 'pm' else root
    missing = sorted(i for i in ids if not (source / i).is_dir())
    if missing:
        raise FileNotFoundError(f'{len(missing)} test objects missing, first: {missing[:3]}')
    common = dict(root=str(root), split=args.split, image_size=tuple(args.img_size), ho=1, ha=1,
                  use_views=1, expand_single_view=True, object_ids=ids)
    if args.dataset == 'pm':
        ds = PartNetMobilityActionsDataset(**common, num_types=2,
            eval_fixed_view_idx=args.view_ids, eval_fixed_frame_list=[0, 2])
        available = {str(Path(tr['jdir']).parent.relative_to(source)) for tr in ds.tr}
    else:
        ds = ACDInferenceDataset(**common)
        available = {f"{tr['data_name']}/{tr['cid']}/{tr['oid']}" for tr in ds.tr}
    skipped = sorted(ids - available)
    if skipped:
        raise ValueError(f'{len(skipped)} test objects have no usable samples, first: {skipped[:3]}')
    return ds


def sample_path(meta, dataset):
    if 'object_key' in meta:
        return Path(meta['object_key']) / meta['views'][0] / f"joint_{int(meta['jid'])}"
    if dataset == 'pm':
        return Path(meta['oid']) / f"joint_{int(meta['jid'])}" / meta['views'][0]
    if dataset == 'acd':
        return Path(meta['data_name']) / meta['oid'] / f"joint_{int(meta['jid'])}"
    return Path(meta['oid']) / f"joint_{int(meta['jid'])}"
