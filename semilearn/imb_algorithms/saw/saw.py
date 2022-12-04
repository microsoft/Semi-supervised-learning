# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from inspect import signature
import torch
import numpy as np


from .utils import SAWWeightsHook, SAWCELoss, SAWConsistencyLoss

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import get_data_loader, IMB_ALGORITHMS


@IMB_ALGORITHMS.register('saw')
class SAW(ImbAlgorithmBase):
    """
        SAW algorithm (https://proceedings.mlr.press/v162/lai22b.html).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.ce_loss = SAWCELoss()
        self.consistency_loss = SAWConsistencyLoss()


    def set_hooks(self):
        super().set_hooks()

        # get ground truth distribution
        y_lb_cnt = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            y_lb_cnt[c] += 1
        y_lb_cnt = torch.from_numpy(np.array(y_lb_cnt))
        num_ulb = len(self.dataset_dict['train_ulb'])

        # add weight hooks
        self.register_hook(SAWWeightsHook(y_lb_cnt=y_lb_cnt, num_ulb=num_ulb),
                           "SAWWeightsHook", "NORMAL")


    def set_dataset(self):
        dataset_dict = super().set_dataset()
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
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


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['x_lb_weights'] = self.x_lb_weights.cpu()
        save_dict['x_ulb_weights'] = self.x_ulb_weights.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.x_lb_weights = checkpoint['x_lb_weights'].cuda(self.gpu)
        self.x_ulb_weights = checkpoint['x_ulb_weights'].cuda(self.gpu)
