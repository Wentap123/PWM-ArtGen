"""Copy a minimal, portable two-view PWM inference dataset (no meshes)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import math
import shutil

from evaluation.inputs import load_ids
from retrieval.database import object_json_path


def read(path):
    return json.loads(path.read_text())


def bbox(data):
    result = {key: data[key] for key in ('base_bbox_cxcywh_norm', 'joint_bbox_cxcywh_norm')}
    if any(not isinstance(v, list) or len(v) != 4 or
           not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
           for v in result.values()):
        raise ValueError('Missing or invalid bbox')
    return result


def plan_object(root, oid, dataset):
    source = root / 'test' / oid if dataset == 'pm' else root / oid
    annotation = read(object_json_path(source))
    nodes = sorted((n for n in annotation['diffuse_tree'] if n['name'] != 'base'), key=lambda n:n['id'])
    graph = {'meta': {'obj_cat': annotation['meta']['obj_cat']}, 'diffuse_tree': [
        {k: n[k] for k in ('id', 'parent', 'name', 'children')} for n in annotation['diffuse_tree']]}
    files, views = {}, {}
    if dataset == 'pm':
        info = read(sorted(source.glob('*_joints_info.json'))[0])
        names = info.get('new_joint_name_list', info.get('joint_name_list'))
        labels = info.get('new_joint_global_id_list', info.get('joint_global_id_list'))
        label_map = dict(zip(names, labels))
        joints = sorted((p for p in source.glob('joint_*') if p.is_dir()), key=lambda p:int(p.name[6:]))
        if len(joints) != len(nodes):
            raise ValueError(f'Joint count mismatch: {oid}')
        for v in (0, 1):
            view = f'view_{v}'
            samples = []
            for joint, node in zip(joints, nodes):
                rgb_dir = joint / 'RGB' / f'view_idx_{v:02d}'
                metadata = read(rgb_dir / 'info/meta.json')
                mask_root = joint / 'Mask' if (joint / 'Mask').is_dir() else joint / 'mask'
                rgb, mask = f'{view}/{joint.name}/rgb.png', f'{view}/{joint.name}/mask.png'
                files[rgb] = rgb_dir / f'frame_000_view_{v:02d}.png'
                files[mask] = mask_root / f'view_idx_{v:02d}' / f'frame_000_view_{v:02d}.png'
                samples.append(dict(joint_id=int(joint.name[6:]), node_id=node['id'],
                                    label_id=label_map[joint.name], rgb=rgb, mask=mask, **bbox(metadata)))
            views[view] = {'samples': samples}
    else:
        def view_samples(original, view):
            boxes = read(source / 'info' / f'{original}_bbox.json')['bboxes']
            rgb, mask = f'{view}/rgb.png', f'{view}/mask.png'
            src_rgb, src_mask = source / 'rgbs' / f'{original}.png', source / 'masks' / f'{original}.png'
            if not src_rgb.is_file() or not src_mask.is_file():
                raise FileNotFoundError(f'{oid}: missing view {original}')
            samples = [dict(joint_id=n['id'], node_id=n['id'], label_id=n['seg_id'],
                            rgb=rgb, mask=mask, **bbox(boxes[str(n['seg_id'])])) for n in nodes]
            return samples, {rgb:src_rgb, mask:src_mask}
        try:
            first, first_files = view_samples('10', 'view_0')
        except (KeyError, ValueError, OSError):
            # If the reference is incomplete, prefer views where the least
            # visible movable part occupies the most pixels.
            import numpy as np
            from PIL import Image
            complete = []
            for candidate in sorted((source / 'rgbs').glob('*.png')):
                try:
                    samples, selected = view_samples(candidate.stem, 'view_0')
                    with Image.open(selected['view_0/mask.png']) as im:
                        mask = np.array(im)
                        resized = np.array(im.resize((256, 256), Image.Resampling.NEAREST))
                    counts = [int(np.count_nonzero(mask == n['seg_id'])) for n in nodes]
                    if (mask.ndim != 2 or not counts or min(counts) == 0 or
                            any(not np.any(resized == n['seg_id']) for n in nodes) or
                            any(sample[key][i] <= 0 for sample in samples
                                for key in ('base_bbox_cxcywh_norm', 'joint_bbox_cxcywh_norm')
                                for i in (2, 3))):
                        continue
                    complete.append((min(counts), candidate.stem))
                except (KeyError, ValueError, OSError):
                    continue
            complete.sort(key=lambda item: (-item[0], item[1]))
            if len(complete) < 2:
                raise ValueError(f'{oid}: fewer than two complete views')
            for index, (_, original) in enumerate(complete[:2]):
                view = f'view_{index}'
                samples, selected = view_samples(original, view)
                files.update(selected)
                views[view] = {'samples': samples}
        else:
            files.update(first_files)
            views['view_0'] = {'samples': first}
            candidates = {}
            for candidate in sorted((source / 'rgbs').glob('*.png')):
                if candidate.stem == '10':
                    continue
                try:
                    samples, selected = view_samples(candidate.stem, 'view_1')
                except (KeyError, ValueError, OSError):
                    continue
                candidates[candidate.stem] = (samples, selected)
            if not candidates:
                raise ValueError(f'{oid}: no complete second view')
            from evaluation.frontal import rank_views
            _, chosen = rank_views(source, candidates, [n['seg_id'] for n in nodes])[0]
            samples, selected = candidates[chosen]
            files.update(selected)
            views['view_1'] = {'samples': samples}
    if not nodes:
        raise ValueError(f'{oid}: no movable parts')
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {'object_id':oid, 'graph':graph, 'views':views}, files


def export(args):
    root, output = args.data_root.resolve(), args.out_dir.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Source and destination must be separate.')
    if output.exists():
        raise FileExistsError(f'Destination already exists; use a fresh directory: {output}')
    ids_path = args.test_ids or ROOT / 'splits' / f'{args.dataset}_test.json'
    ids = load_ids(ids_path, args.dataset)
    plans, errors = [], []
    for oid in sorted(ids):
        try:
            plans.append(plan_object(root, oid, args.dataset))
        except Exception as error:
            errors.append(f'{oid}: {error}')
    if errors:
        raise ValueError('No files copied. Preflight failures:\n' + '\n'.join(errors))
    size = sum(p.stat().st_size for _, files in plans for p in files.values())
    print(f'{args.dataset}: {len(plans)} objects, 2 views, {size / 1024**2:.1f} MiB images', flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True)
    hashes = {}
    for index, (obj, files) in enumerate(plans):
        dest = output / obj['object_id']
        for relative, source in files.items():
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise IOError(f'Copy verification failed: {target}')
            hashes[target.relative_to(output).as_posix()] = digest
        for filename, data in [('graph.json', obj.pop('graph')), ('views.json', obj)]:
            path = dest / filename
            path.write_text(json.dumps(data, indent=2) + '\n')
            hashes[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        if (index + 1) % 20 == 0:
            print(f'Copied {index + 1}/{len(plans)} objects', flush=True)
    (output / 'test_ids.json').write_text(json.dumps({'dataset': args.dataset, 'split': 'test', 'object_ids': sorted(ids)}, indent=2) + '\n')
    (output / 'checksums.json').write_text(json.dumps(hashes, indent=2) + '\n')
    # Written last: loaders reject partially copied datasets.
    (output / 'dataset.json').write_text(json.dumps({
        'format':'pwm-two-view-v1', 'dataset':args.dataset, 'objects':len(ids),
        'views':['view_0','view_1'],
        'images_bytes':size, 'complete':True}, indent=2) + '\n')
    print(f'Complete: {output}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['pm','acd'])
    ap.add_argument('--data_root', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--test_ids', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    try:
        export(args)
    except (ValueError, OSError) as error:
        ap.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
