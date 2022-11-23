# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F

from .utils import EffectiveDistribution, LogitsAdjCELoss

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import get_data_loader, get_dataset, IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('simis')
class SimiS(ImbAlgorithmBase):
    """
        SimiS algorithm (https://arxiv.org/abs/2211.11086).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - simis_la (bool):
                flag of using logits adjustment
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        args.include_lb_to_ulb = True
        super(SimiS, self).__init__(args, net_builder, tb_log, logger, **kwargs)
        
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(lb_class_dist / lb_class_dist.sum()).float()
        self.lb_class_dist = self.lb_class_dist.to(self.gpu)

        if args.simis_la:
            self.ce_loss = LogitsAdjCELoss(lb_class_dist=self.lb_class_dist)

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(EffectiveDistribution(), "RefineLabelSetHook", "NORMAL")

    def set_dataset(self):
        dataset_dict = super().set_dataset()
        eval_ulb = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, False)['train_ulb']
        dataset_dict['eval_ulb'] = copy.deepcopy(eval_ulb)
        dataset_dict['eval_ulb'].is_ulb = False
        return dataset_dict
    
    def set_data_loader(self):
        loader_dict = super().set_data_loader()
        
        # add unlabeled evaluation data loader
        loader_dict['eval_ulb'] = get_data_loader(self.args,
                                                  self.dataset_dict['eval_ulb'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)

        return loader_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--simis_la', str2bool, False),
        ]
