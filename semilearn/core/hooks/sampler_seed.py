# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/sampler_seed.py

from .hook import Hook

from semilearn.datasets import DistributedSampler

class DistSamplerSeedHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    def before_train_epoch(self, algorithm):
        if isinstance(algorithm.loader_dict['train_lb'].sampler, DistributedSampler):
            algorithm.loader_dict['train_lb'].sampler.set_epoch(algorithm.epoch)
        if isinstance(algorithm.loader_dict['train_ulb'].sampler, DistributedSampler):
            algorithm.loader_dict['train_ulb'].sampler.set_epoch(algorithm.epoch)