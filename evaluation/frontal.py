"""Deterministic 2D proxy for similarity to a known frontal reference view.

No predictions, 3D GT geometry or evaluation scores are used for view selection.
Center/scale normalization preserves aspect ratio; labels preserve part identity.
"""
from pathlib import Path
import numpy as np
from PIL import Image


def normalized_labels(path, size=160):
    with Image.open(path) as image:
        labels = np.array(image)
    if labels.ndim != 2:
        raise ValueError(f'Expected integer label mask: {path}')
    ys, xs = np.nonzero(labels)
    if not len(xs):
        raise ValueError(f'Empty mask: {path}')
    crop = labels[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.int32)
    h, w = crop.shape
    scale = (size - 4) / max(h, w)
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.array(Image.fromarray(crop).resize((width, height), Image.Resampling.NEAREST))
    canvas = np.zeros((size, size), dtype=np.int32)
    y, x = (size-height)//2, (size-width)//2
    canvas[y:y+height, x:x+width] = resized
    return canvas


def similarity(reference, candidate, labels):
    def iou(a, b):
        union = np.count_nonzero(a | b)
        return float(np.count_nonzero(a & b) / union) if union else 0.0
    silhouette = iou(reference != 0, candidate != 0)
    parts = [iou(reference == label, candidate == label) for label in labels]
    return 0.25 * silhouette + 0.75 * float(np.mean(parts))


def rank_views(source, names, labels):
    source = Path(source)
    reference = normalized_labels(source / 'masks/10.png')
    return sorted(((similarity(reference, normalized_labels(source / 'masks' / f'{name}.png'), labels), name)
                   for name in names), key=lambda pair: (-pair[0], pair[1]))
