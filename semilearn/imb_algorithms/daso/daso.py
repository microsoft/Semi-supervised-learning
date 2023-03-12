# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import queue
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from .utils import DASOFeatureQueue, DASOPseudoLabelingHook

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('daso')
class DASO(ImbAlgorithmBase):
    """
        DASO algorithm (https://arxiv.org/abs/2106.05682).

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
        self.imb_init(T_proto=args.daso_T_proto, T_dist=args.daso_T_dist, daso_queue_len=args.daso_queue_len,
                      interp_alpha=args.daso_interp_alpha, with_dist_aware=args.daso_with_dist_aware, assign_loss_ratio=args.daso_assign_loss_ratio,
                      num_pl_dist_iter=args.daso_num_pl_dist_iter, num_pretrain_iter=args.daso_num_pretrain_iter)
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        
        # get queue
        self.queue = DASOFeatureQueue(num_classes=self.num_classes, 
                                      feat_dim=self.model.num_features, 
                                      queue_length=self.daso_queue_len)
        self.similarity_fn = nn.CosineSimilarity(dim=2)


    def imb_init(self, T_proto=0.05, T_dist=1.5, 
                       daso_queue_len=256, interp_alpha=0.3, with_dist_aware=True, assign_loss_ratio=1.0, 
                       num_pl_dist_iter=100, num_pretrain_iter=5120):
        self.T_proto = T_proto
        self.T_dist = T_dist
        self.daso_queue_len = daso_queue_len
        self.interp_alpha = interp_alpha
        self.lambda_f = assign_loss_ratio
        self.with_dist_aware = with_dist_aware
        self.num_pl_dist_iter = num_pl_dist_iter
        self.num_pretrain_iter = num_pretrain_iter

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(DASOPseudoLabelingHook(num_classes=self.num_classes, T_dist=self.T_dist, with_dist_aware=self.with_dist_aware, interp_alpha=self.interp_alpha), 
                           "PseudoLabelingHook", "LOWEST")

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)


    def train_step(self, *args, **kwargs):
        # push memory queue using ema model
        self.ema.apply_shadow()
        with torch.no_grad():
            x_lb, y_lb = kwargs['x_lb'], kwargs['y_lb']
            feats_x_lb = self.model(x_lb)['feat']
            self.queue.enqueue(feats_x_lb.clone().detach(), y_lb.clone().detach())
        self.ema.restore()

        # forward through loop
        out_dict, log_dict = super().train_step(*args, **kwargs)

        if self.it + 1 < self.num_pretrain_iter:
            # get core algorithm output
            return out_dict, log_dict 
        
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]

        # compute semantic labels
        prototypes = self.queue.prototypes  # (K, D)

        with torch.no_grad():
            # similarity between weak features and prototypes  (B, K)
            sim_w = self.similarity_fn(feats_x_ulb_w.unsqueeze(1), prototypes.unsqueeze(0)) / self.T_proto
            prob_sim_w = sim_w.softmax(dim=1)
        self.probs_sim = prob_sim_w.detach()

        # compute soft loss
        # similarity between strong features and prototypes  (B, K)
        sim_s = self.similarity_fn(feats_x_ulb_s.unsqueeze(1), prototypes.unsqueeze(0)) / self.T_proto
        assign_loss = self.ce_loss(sim_s, prob_sim_w, reduction='mean')

        # add assign loss 
        out_dict['loss'] += self.lambda_f * assign_loss
        log_dict['train/assign_loss'] = assign_loss.item()
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        # save_dict['queue_bank'] = self.queue.bank
        save_dict['queue_prototypes'] = self.queue.prototypes.cpu()
        save_dict['pl_list'] = self.hooks_dict['PseudoLabelingHook'].pseudo_label_list
        save_dict['pl_dist'] = self.hooks_dict['PseudoLabelingHook'].pseudo_label_dist
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        # self.queue.bank = checkpoint['queue_bank'] 
        self.queue.prototypes = checkpoint['queue_prototypes'] 
        self.hooks_dict['PseudoLabelingHook'].pseudo_label_list = checkpoint['pl_list']
        self.hooks_dict['PseudoLabelingHook'].pseudo_label_dist = checkpoint['pl_dist']
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--daso_queue_len', int, 256),
            SSL_Argument('--daso_T_proto', float, 0.05),
            SSL_Argument('--daso_T_dist', float, 1.5),
            SSL_Argument('--daso_interp_alpha', float, 0.5),
            SSL_Argument('--daso_with_dist_aware', str2bool, True),
            SSL_Argument('--daso_assign_loss_ratio', float, 1.0),
            SSL_Argument('--daso_num_pl_dist_iter', int, 100),
            SSL_Argument('--daso_num_pretrain_iter', int, 5120),
        ]        


    
    