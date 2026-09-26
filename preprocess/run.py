"""Preprocessing worker, executed in the selected segmentation environment."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

from .parts import bbox, category_of, sam_parts, sam3_parts, validate_graph


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def export_inputs(graph, labels, rgba, root, backend, category):
    from .soft_mask import make_soft_mask
    root = Path(root)
    view = root / 'data/test' / category / 'input/view_id_00'
    graph.setdefault('meta', {})['obj_cat'] = category
    write_json(view / 'process/graph_renum.json', graph)
    Image.fromarray(labels.astype(np.int32)).save(view / 'process/mask.png')
    base_mask = np.asarray(rgba.getchannel('A')) > 0 if backend == 'sam3' else labels != 0
    base_bbox = bbox(base_mask, backend)
    rgb = np.asarray(rgba.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
    scaled_labels = np.asarray(Image.fromarray(labels.astype(np.int32)).resize((256, 256), Image.Resampling.NEAREST))
    for node in graph['diffuse_tree'][1:]:
        label = node['seg_ids'][0]
        joint_bbox = bbox(labels == label, backend)
        binary = np.repeat((scaled_labels == label)[..., None], 3, axis=-1).astype(np.uint8)
        alpha = make_soft_mask(binary, do_close=True, close_kernel=3, kernel=5, gauss_sigma=1.2)
        masked = (rgb * alpha).astype(np.uint8)
        joint_dir = view / f"joint_{node['id']}"
        joint_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(joint_dir / 'frame_000_view_00.png')
        Image.fromarray(masked).save(joint_dir / 'frame_000_view_00_masked.png')
        write_json(joint_dir / 'mask_renum_bbox.json', {
            'id': node['id'], 'base_bbox_cxcywh_norm': base_bbox,
            'joint_bbox_cxcywh_norm': joint_bbox})
    return {'data_root': str(root / 'data'), 'category': category,
            'object_id': 'input', 'num_joints': len(graph['diffuse_tree']) - 1}


def prepare(config, output):
    from .backends import prepare_image, segment_sam, segment_sam3, region_visualization
    from .graph_api import predict_graph, assign_regions
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    needs_graph = not (output / 'graph.json').exists() and not config.get('graph_json')
    needs_ids = (config['seg_backend'] == 'sam' and not (output / 'assignments.json').exists()
                 and not config.get('assignments_json'))
    if (needs_graph or needs_ids) and not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('Set OPENAI_API_KEY, or supply cached graph/assignment JSON before running preprocessing.')
    rgba_path = output / 'image.png'
    if not rgba_path.exists():
        prepare_image(config['image'], config.get('rmbg_model'), config['device']).save(rgba_path)
    rgba = Image.open(rgba_path).convert('RGBA')
    graph_path = output / 'graph.json'
    if not graph_path.exists():
        graph = (read_json(config['graph_json']) if config.get('graph_json') else
                 predict_graph(rgba.convert('RGB'), config['graph_model']))
        validate_graph(graph)
        write_json(graph_path, graph)
    graph = read_json(graph_path)
    validate_graph(graph)
    category = category_of(graph, config.get('category'))
    backend = config['seg_backend']
    if backend == 'sam':
        groups_path = output / 'sam_regions.npy'
        if not groups_path.exists():
            groups = segment_sam(rgba, config['seg_checkpoint'], config['device'])
            np.save(groups_path, groups, allow_pickle=False)
        groups = np.load(groups_path, allow_pickle=False)
        if not np.any(groups >= 0):
            raise ValueError('SAM found no foreground regions.')
        visualization = region_visualization(groups)
        visualization.save(output / 'regions.png')
        assigned_path = output / 'assignments.json'
        if not assigned_path.exists():
            assigned = (read_json(config['assignments_json']) if config.get('assignments_json') else
                        assign_regions(visualization, graph, config['graph_model']))
            write_json(assigned_path, assigned)
        graph, labels = sam_parts(graph, read_json(assigned_path), groups)
    else:
        masks_path = output / 'sam3_masks.npz'
        if not masks_path.exists():
            masks = segment_sam3(rgba, graph, config['seg_checkpoint'], config['device'], config['confidence'])
            np.savez_compressed(masks_path, **masks)
        with np.load(masks_path, allow_pickle=False) as masks:
            graph, labels = sam3_parts(graph, dict(masks))
        region_visualization(labels - 1).save(output / 'regions.png')
    result = export_inputs(graph, labels, rgba, output, backend, category)
    write_json(output / 'prepared.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    manifest = read_json(args.run / 'run.json')
    prepare(manifest['config'], args.run / 'preprocess')


if __name__ == '__main__':
    try:
        main()
    except ImportError as error:
        raise SystemExit(f'Missing preprocessing dependency: {error}. Follow preprocess/NOTICE.md and the SAM/SAM3 official environment setup.')
