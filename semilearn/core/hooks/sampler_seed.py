# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/sampler_seed.py

from torch.utils.data import DataLoader
from .hook import Hook
from semilearn.datasets import DistributedSampler

class DistSamplerSeedHook(Hook):
    """
    Distributed sampler seed Hook

    update the samples' epoch in data loader
    """
    def before_train_epoch(self, algorithm):
        for name, dataloader in algorithm.loader_dict.items():
            if not isinstance(dataloader, DataLoader):
                continue

            if isinstance(dataloader.sampler, DistributedSampler):
                algorithm.loader_dict[name].sampler.set_epoch(algorithm.epoch)