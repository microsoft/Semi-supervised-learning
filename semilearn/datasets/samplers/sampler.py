# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import random


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, num_samples=None, **kwargs):
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.total_size = num_samples
        assert num_samples % self.num_replicas == 0, f'{num_samples} samples cant' \
                                                     f'be evenly distributed among {num_replicas} devices.'
        self.num_samples = int(num_samples // self.num_replicas)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        n = len(self.dataset)
        n_repeats = self.total_size // n
        n_remain = self.total_size % n
        indices = [torch.randperm(n, generator=g) for _ in range(n_repeats)]
        indices.append(torch.randperm(n, generator=g)[:n_remain])
        indices = torch.cat(indices, dim=0).tolist()

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class WeightedDistributedSampler(DistributedSampler):
    def __init__(self, weights, dataset, num_replicas=None, rank=None, num_samples=None, replacement=False):
        super().__init__(dataset, num_replicas, rank, num_samples)
        self.replacement = replacement
        self.sample_weights = self.get_sample_weights(weights)
    
    def get_sample_weights(self, weights):
        targets = self.dataset.targets
        sample_weight = torch.tensor([weights[t] for t in targets])
        return sample_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        n = len(self.dataset)
        n_repeats = self.total_size // n
        n_remain = self.total_size % n

        indices = [torch.multinomial(self.sample_weights, n, generator=g, replacement=self.replacement) for _ in range(n_repeats)]
        indices.append(torch.multinomial(self.sample_weights, n, generator=g, replacement=self.replacement)[:n_remain])
        indices = torch.cat(indices, dim=0).tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class ImageNetDistributedSampler(DistributedSampler):
    def __init__(self, dataset_idx, num_replicas=None, rank=None, num_samples=None):
        """
        """
        super().__init__(dataset=dataset_idx, num_replicas=num_replicas, rank=rank, num_samples=num_samples)
        if isinstance(dataset_idx, list):
            self.dataset = np.array(dataset_idx)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        n = len(self.dataset)
        n_repeats = self.total_size // n
        n_remain = self.total_size % n
        indices = [torch.randperm(n, generator=g) for _ in range(n_repeats)]
        indices.append(torch.randperm(n, generator=g)[:n_remain])
        indices = torch.cat(indices, dim=0).tolist()

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # indices is a proxy index for sampling labeled / unlabeled index from dataset_idx
        return iter(self.dataset[indices])


name2sampler = {
    'RandomSampler': DistributedSampler, 
    'WeightedRandomSampler': WeightedDistributedSampler}
