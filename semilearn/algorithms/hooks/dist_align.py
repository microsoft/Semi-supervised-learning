# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import concat_all_gather


class DistAlignEMAHook(Hook):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target)    
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, algorithm, ulb_probs, lb_probs=None):
        # update queue
        self.update_p(algorithm, ulb_probs, lb_probs)

        # dist align
        ulb_probs_aligned = ulb_probs * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        ulb_probs_aligned = ulb_probs_aligned / ulb_probs_aligned.sum(dim=-1, keepdim=True)
        return ulb_probs_aligned
    

    @torch.no_grad()
    def update_p(self, algorithm, ulb_probs, lb_probs):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(ulb_probs.device)

        if algorithm.distributed and algorithm.world_size > 1:
            if lb_probs is not None:
                lb_probs = concat_all_gather(lb_probs)
            ulb_probs = concat_all_gather(ulb_probs)

        ulb_probs = ulb_probs.detach()
        if self.p_model == None:
            self.p_model = torch.mean(ulb_probs, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(ulb_probs, dim=0) * (1 - self.m)

        if self.update_p_target:
            assert lb_probs is not None
            self.p_target = self.p_target * self.m + torch.mean(lb_probs, dim=0) * (1 - self.m)
    
    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes, )) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes, ))/ self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
        
        return update_p_target, p_target