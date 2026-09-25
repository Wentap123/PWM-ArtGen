"""Pure graph/mask normalization shared by both single-image backends."""
from copy import deepcopy
import re
import numpy as np

CATEGORIES = ('StorageFurniture', 'Table', 'Refrigerator', 'Oven', 'Microwave',
              'WashingMachine', 'Dishwasher')


def category_of(graph, override=None):
    raw = override or graph.get('meta', {}).get('obj_cat')
    if not raw:
        match = re.search(r'articulated parts (?:in|of) (?:a|an) (.*?),',
                          graph.get('original_response', ''), flags=re.I)
        raw = match.group(1) if match else ''
    key = re.sub(r'[^a-z]', '', raw.lower())
    aliases = {re.sub(r'[^a-z]', '', c.lower()): c for c in CATEGORIES}
    aliases.update(washer='WashingMachine', storagecabinet='StorageFurniture')
    if key not in aliases:
        raise ValueError(f'Unknown category {raw!r}; supply --category from {CATEGORIES}.')
    return aliases[key]


def flatten_tree(tree):
    nodes = []
    def visit(value, parent):
        if not isinstance(value, dict) or len(value) != 1:
            raise ValueError('Each graph node must contain exactly one part name.')
        name, children = next(iter(value.items()))
        name = 'door' if name == 'lid' else name
        index = len(nodes)
        node = dict(id=index, parent=parent, name=name, children=[])
        nodes.append(node)
        if not isinstance(children, list):
            raise ValueError('Graph children must be a list.')
        for child in children:
            node['children'].append(visit(child, index))
        return index
    visit(tree, -1)
    graph = {'diffuse_tree': nodes}
    validate_graph(graph)
    return graph


def validate_graph(graph):
    nodes = graph.get('diffuse_tree', [])
    if not nodes or nodes[0].get('name') != 'base' or nodes[0].get('parent') != -1:
        raise ValueError('Expected a single base root.')
    ids = [n.get('id') for n in nodes]
    if len(ids) != len(set(ids)) or any(not isinstance(i, int) for i in ids):
        raise ValueError('Graph node IDs must be distinct integers.')
    if nodes[0].get('children') != ids[1:]:
        raise ValueError('Single-image graphs must attach every movable part directly to base.')
    for node in nodes[1:]:
        if node.get('name') not in ('door', 'drawer') or node.get('parent') != ids[0] or node.get('children'):
            raise ValueError('Only door/drawer children are supported; merge handles into parents.')
    return nodes


def normalize_nodes(graph):
    result = deepcopy(graph)
    nodes = result['diffuse_tree']
    for i, node in enumerate(nodes):
        node.update(id=i, parent=-1 if i == 0 else 0,
                    children=list(range(1, len(nodes))) if i == 0 else [])
    validate_graph(result)
    if len(nodes) < 2:
        raise ValueError('No visible movable parts survived segmentation.')
    return result


def sam_parts(graph, assignments, group_ids):
    """Match instance IDs, merge regions, renumber, and apply legacy containment filtering."""
    result = deepcopy(graph)
    nodes = validate_graph(result)
    tree = assignments.get('json', assignments)
    base = tree.get('base') if isinstance(tree, dict) else None
    if not isinstance(base, dict):
        raise ValueError('Invalid region-assignment response.')
    children = base.get('children', [])
    if len(children) != len(nodes) - 1:
        raise ValueError('Region assignment changed the predicted part count.')
    entries = [base]
    for node, child in zip(nodes[1:], children):
        if not isinstance(child, dict) or list(child) != [node['name']]:
            raise ValueError('Region assignment changed part names or ordering.')
        entries.append(child[node['name']])
    mask = np.zeros(group_ids.shape, dtype=np.int32)
    owners = set()
    available = set(int(i) for i in np.unique(group_ids) if i >= 0)
    targets = []
    for entry in entries:
        ids = entry.get('ids', [])
        if any(not isinstance(i, int) or i not in available for i in ids):
            raise ValueError('Region assignment references a nonexistent region.')
        if owners.intersection(ids) or len(set(ids)) != len(ids):
            raise ValueError('A segmentation region was assigned more than once.')
        owners.update(ids)
        targets.append(min(ids) + 1 if ids else None)
    # Use the unmodified source array for every remap, avoiding ID collisions.
    for node, entry, target in zip(nodes, entries, targets):
        node['seg_ids'] = [target] if target is not None else []
        for old in entry.get('ids', []):
            mask[group_ids == old] = target
    remap = {old: i + 1 for i, old in enumerate(sorted(set(targets) - {None}))}
    renumbered = np.zeros_like(mask)
    for old, new in remap.items():
        renumbered[mask == old] = new
    for node in nodes:
        node['seg_ids'] = [remap[i] for i in node['seg_ids']]
    mask = renumbered
    # The executed legacy filter uses containment (ratio >= 1), not its stale 10% comment.
    candidates = [n for n in nodes[1:] if n['seg_ids']]
    votes = set()
    def bounds(node):
        ys, xs = np.where(mask == node['seg_ids'][0])
        return (xs.min(), ys.min(), xs.max(), ys.max())
    for i, first in enumerate(candidates):
        a = bounds(first)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        for second in candidates[i+1:]:
            b = bounds(second)
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if min(area_a, area_b) > 0 and inter > 0 and (inter >= area_a or inter >= area_b):
                votes.add((first if area_a > area_b else second)['seg_ids'][0])
    base_id = nodes[0]['seg_ids'][0] if nodes[0]['seg_ids'] else next(
        (n['seg_ids'][0] for n in candidates if n['seg_ids'][0] in votes), None)
    if base_id is not None:
        nodes[0]['seg_ids'] = [base_id]
        for value in votes:
            mask[mask == value] = base_id
    kept = [n for n in candidates if n['seg_ids'][0] not in votes]
    kept.sort(key=lambda n: 0 if n['name'] == 'door' else 1)
    result['diffuse_tree'] = [nodes[0]] + kept
    return normalize_nodes(result), mask


def sam3_parts(graph, masks_by_name):
    """Keep the original SAM3 ordering: base, drawers, doors; later masks own overlaps."""
    entries = []
    for name in ('drawer', 'door'):
        masks = masks_by_name.get(name, [])
        for mask in masks:
            entries.append((name, np.asarray(mask) > 0.5))
    if not entries:
        raise ValueError('SAM3 found no movable parts.')
    labels = np.zeros(entries[0][1].shape, dtype=np.int32)
    nodes = [dict(id=0, parent=-1, name='base', children=[], seg_ids=[1])]
    for i, (name, mask) in enumerate(entries, 1):
        if mask.shape != labels.shape:
            raise ValueError('SAM3 returned inconsistent mask dimensions.')
        labels[mask] = i + 1
        nodes.append(dict(id=i, parent=0, name=name, children=[], seg_ids=[i+1]))
    if any(not (labels == n['seg_ids'][0]).any() for n in nodes[1:]):
        raise ValueError('SAM3 produced an empty or fully occluded part mask.')
    result = deepcopy(graph)
    result['diffuse_tree'] = nodes
    return normalize_nodes(result), labels


def bbox(mask, backend):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('Cannot compute a bounding box for an empty mask.')
    height, width = mask.shape
    offset = 0.5 if backend == 'sam3' else 0.0
    return [(float(xs.min()+xs.max())/2 + offset)/width,
            (float(ys.min()+ys.max())/2 + offset)/height,
            float(xs.max()-xs.min()+1)/width, float(ys.max()-ys.min()+1)/height]
