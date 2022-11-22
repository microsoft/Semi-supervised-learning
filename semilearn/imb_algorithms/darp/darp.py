
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from inspect import signature

from .utils import DARPPseudoLabelingHook
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@IMB_ALGORITHMS.register('darp')
class DARP(ImbAlgorithmBase):
    """
        DARP algorithm (https://arxiv.org/abs/2007.08844).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - darp_warmup_epochs (int):
                warmup epochs for darp
            - darp_alpha (float):
                alpha parameter for darp
            - darp_iter_T (int):
                darp optimization T
            - darp_num_refine_iter
                darp refine iterations 
    """
    def __init__(self, args, **kwargs):
        self.imb_init(warmup_epochs=args.darp_warmup_epochs, alpha=args.darp_alpha, iter_T=args.darp_iter_T, num_refine_iter=args.darp_num_refine_iter)
        super().__init__(args, **kwargs)
        
    def imb_init(self, warmup_epochs=200, alpha=2.0, iter_T=10, num_refine_iter=10):
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.iter_T = iter_T
        self.num_refine_iter = num_refine_iter

    def set_hooks(self):
        super().set_hooks()

        # get ground truth distribution
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        lb_class_dist = lb_class_dist / lb_class_dist.sum()
        target_disb = lb_class_dist * len(self.dataset_dict['train_ulb']) / lb_class_dist.sum()

        # reset PseudoLabelingHook hook
        self.register_hook(DARPPseudoLabelingHook(warmup_epochs=self.warmup_epochs, alpha=self.alpha, iter_T=self.iter_T,
                                                  num_refine_iter=self.num_refine_iter, dataset_len=len(self.dataset_dict['train_ulb']),
                                                  num_classes=self.num_classes, target_disb=target_disb),
                           "PseudoLabelingHook",)

    def process_batch(self, **kwargs):
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys()) + ['idx_ulb']
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):
        self.idx_ulb = kwargs['idx_ulb']
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        if 'idx_ulb' not in input_args:
            kwargs.pop('idx_ulb')
        return super().train_step(*args, **kwargs)
    
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['pseudo_orig'] = self.hooks_dict['PseudoLabelingHook'].pseudo_orig
        save_dict['pseudo_refine'] = self.hooks_dict['PseudoLabelingHook'].pseudo_refine
        return save_dict
        
    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['PseudoLabelingHook'].pseudo_orig = checkpoint['pseudo_orig']
        self.hooks_dict['PseudoLabelingHook'].pseudo_refine = checkpoint['pseudo_refine']


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--darp_warmup_epochs', int, 200),
            SSL_Argument('--darp_alpha', float, 2.0),
            SSL_Argument('--darp_iter_T', int, 10),
            SSL_Argument('--darp_num_refine_iter', int, 10),
        ]
