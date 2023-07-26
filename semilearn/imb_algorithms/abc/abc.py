# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


class ABCNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # auxiliary classifier
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@IMB_ALGORITHMS.register('abc')
class ABC(ImbAlgorithmBase):
    """
        ABC algorithm (https://arxiv.org/abs/2110.10368).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - abc_p_cutoff (`float`):
                threshold for the auxiliary classifier
            - abc_loss_ratio (`float`):
                loss ration for auxiliary classifier
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(abc_p_cutoff=args.abc_p_cutoff, abc_loss_ratio=args.abc_loss_ratio)

        super(ABC, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        
        # TODO: better ways
        self.model = ABCNet(self.model, num_classes=self.num_classes)
        self.ema_model = ABCNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

    def imb_init(self, abc_p_cutoff=0.95, abc_loss_ratio=1.0):
        self.abc_p_cutoff = abc_p_cutoff
        self.abc_loss_ratio = abc_loss_ratio

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):
        
        out_dict, log_dict = super().train_step(*args, **kwargs)

        # get features
        feats_x_lb = out_dict['feat']['x_lb']
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]
        
        # get logits
        logits_x_lb = self.model.module.aux_classifier(feats_x_lb)
        logits_x_ulb_s = self.model.module.aux_classifier(feats_x_ulb_s)
        with torch.no_grad():
            logits_x_ulb_w = self.model.module.aux_classifier(feats_x_ulb_w)

        # compute abc loss using logits_aux from dict
        abc_loss = self.compute_abc_loss(
            logits_x_lb=logits_x_lb, 
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w,
            logits_x_ulb_s=logits_x_ulb_s
            )
        out_dict['loss'] += self.abc_loss_ratio * abc_loss 
        log_dict['train/abc_loss'] = abc_loss.item()

        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits_aux', return_logits=return_logits)

    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()
    
    def compute_abc_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]
        
        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)

        # compute labeled abc loss
        mask_lb = self.bernouli_mask(self.lb_class_dist[y_lb])
        abc_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

        # compute unlabeled abc loss
        with torch.no_grad():
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask_ulb_1 = max_probs.ge(self.abc_p_cutoff).to(logits_x_ulb_w.dtype)
            ulb_class_dist = 1 - (self.epoch / self.epochs) * (1 - self.lb_class_dist)
            mask_ulb_2 = self.bernouli_mask(ulb_class_dist[y_ulb])
            mask_ulb = mask_ulb_1 * mask_ulb_2
    
        abc_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            abc_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
        
        abc_loss = abc_lb_loss + abc_ulb_loss
        return abc_loss


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--abc_p_cutoff', float, 0.95),
            SSL_Argument('--abc_loss_ratio', float, 1.0),
        ]        
