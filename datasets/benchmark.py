"""Portable PM/ACD observations with identical two-view directory naming."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from preprocess.soft_mask import make_soft_mask


class BenchmarkDataset(Dataset):
    low_dim_dim, action_dim, ha = 8, 20, 1

    def __init__(self, root, dataset, object_ids, view_ids=(0, 1), image_size=(256, 256)):
        self.root, self.image_size = Path(root), tuple(image_size)
        manifest = json.loads((self.root / 'dataset.json').read_text())
        if manifest.get('format') != 'pwm-two-view-v1' or not manifest.get('complete') or manifest['dataset'] != dataset:
            raise ValueError('Incomplete or incompatible exported dataset.')
        self.samples = []
        for oid in sorted(object_ids):
            obj_dir = self.root / oid
            views = json.loads((obj_dir / 'views.json').read_text())['views']
            for v in view_ids:
                view = f'view_{v}'
                for sample in views[view]['samples']:
                    for key in ('rgb', 'mask'):
                        path = (obj_dir / sample[key]).resolve()
                        if not path.is_relative_to(obj_dir.resolve()) or not path.is_file():
                            raise ValueError(f'Invalid {key} path: {path}')
                    self.samples.append((oid, view, sample))
        if not self.samples:
            raise ValueError('No benchmark samples selected.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        oid, view, record = self.samples[index]
        directory = self.root / oid
        with Image.open(directory / record['rgb']) as im:
            rgb = np.array(im.convert('RGB').resize(self.image_size, Image.Resampling.BILINEAR))
        with Image.open(directory / record['mask']) as im:
            if im.mode not in ('P', 'I', 'L'):
                im = im.convert('L')
            mask = np.array(im.resize(self.image_size, Image.Resampling.NEAREST)) == record['label_id']
        alpha = make_soft_mask(np.repeat(mask[..., None], 3, axis=-1).astype(np.uint8),
                               do_close=True, close_kernel=3, kernel=5, gauss_sigma=1.2)
        masked = (rgb * alpha).astype(np.uint8)
        low = np.array(record['base_bbox_cxcywh_norm'] + record['joint_bbox_cxcywh_norm'], dtype=np.float32)
        return {'obs': {'rgb_00':torch.from_numpy(rgb[None]), 'rgb_01':torch.from_numpy(masked[None]),
                        'low_dim':torch.from_numpy(low[None])},
                'meta': {'object_key':oid, 'oid':oid.split('/')[-1], 'jid':record['joint_id'],
                         'node_id':record['node_id'], 'views':[view]}}
