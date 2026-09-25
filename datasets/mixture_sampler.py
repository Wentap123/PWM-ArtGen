import math, numpy as np, torch
from torch.utils.data import Sampler

class InterleavedConcatSampler(Sampler[int]):

    def __init__(self, len_a: int, len_b: int, seed: int = 0, shuffle: bool = True, drop_last: bool = False):
        self.len_a, self.len_b = int(len_a), int(len_b)
        assert self.len_a >= 0 and self.len_b >= 0
        self.total = self.len_a + self.len_b
        assert self.total > 0
        self.seed, self.shuffle, self.drop_last = int(seed), bool(shuffle), bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _build_full_order(self, base_seed: int):
        rng = np.random.RandomState(base_seed ^ self.epoch ^ 0x9E3779B1)
        idx_a = np.arange(self.len_a, dtype=np.int64)
        idx_b = np.arange(self.len_b, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(idx_a); rng.shuffle(idx_b)

        w_a, w_b, cw_a, cw_b = self.len_a, self.len_b, 0, 0
        pa = pb = 0
        offset_b = self.len_a
        order = np.empty(self.total, dtype=np.int64)
        for t in range(self.total):
            cw_a += w_a; cw_b += w_b
            if cw_a >= cw_b:
                order[t] = idx_a[pa]; pa += 1; cw_a -= self.total
            else:
                order[t] = offset_b + idx_b[pb]; pb += 1; cw_b -= self.total
        return order

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        base_seed = self.seed if wi is None else wi.seed
        order = self._build_full_order(base_seed)

        world_size, rank = 1, 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        if world_size > 1:
            if self.drop_last:
                num = (len(order) // world_size) * world_size
                order = order[:num]
                per = num // world_size
                shard = order[rank*per : (rank+1)*per]
            else:
                shard = order[rank::world_size]
        else:
            shard = order
        return iter(shard.tolist())

    def __len__(self):
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        return (self.total // world_size) if self.drop_last else math.ceil(self.total / world_size)
