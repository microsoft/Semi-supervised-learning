# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from semilearn.core.hooks import Hook
from semilearn.core.criterions import CELoss, ConsistencyLoss


def effective_weights(y_cnt):
    N = torch.sum(y_cnt) / len(y_cnt) # Originally N is a hyperparameter
    beta = (N - 1) / N
    effective_num = 1.0 - torch.pow(beta, y_cnt)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(y_cnt)
    return weights


class SAWWeightsHook(Hook):
    def __init__(self, y_lb_cnt, num_ulb) -> None:
        super().__init__()
        self.y_lb_cnt = y_lb_cnt
        self.num_ulb = num_ulb
    
    def before_run(self, algorithm):
        # initialize x_lb_weights and x_ulb_weights
        algorithm.x_lb_weights = effective_weights(self.y_lb_cnt).cuda(algorithm.gpu)
        algorithm.print_fn("labeled data class weights: {}".format(algorithm.x_lb_weights))

        y_ulb_cnt = np.array([self.num_ulb / len(self.y_lb_cnt) for _ in range(len(self.y_lb_cnt))]) 
        y_ulb_cnt = torch.from_numpy(y_ulb_cnt) 
        algorithm.x_ulb_weights = effective_weights(y_ulb_cnt).cuda(algorithm.gpu)
        algorithm.print_fn("unlabeled data class weights: {}".format(algorithm.x_ulb_weights))

    def before_train_epoch(self, algorithm):
        algorithm.ce_loss.set_weights(algorithm.x_lb_weights)
        algorithm.consistency_loss.set_weights(algorithm.x_ulb_weights)

    def after_train_epoch(self, algorithm):
        # evaluate on unlabeled data
        logits_x_ulb = algorithm.evaluate('eval_ulb', return_logits=True)['eval_ulb/logits']
        pred_x_ulb = torch.from_numpy(logits_x_ulb).argmax(dim=-1)
        
        # convert to one_hot
        pred_x_ulb = F.one_hot(pred_x_ulb, num_classes=algorithm.num_classes)
        
        # get pseudo label counts 
        pl_x_ulb_counts = pred_x_ulb.sum(dim=0)
        pl_x_ulb_counts = torch.maximum(pl_x_ulb_counts, torch.ones_like(pl_x_ulb_counts))

        # compute weights and update to algorithm
        algorithm.x_ulb_weights = effective_weights(pl_x_ulb_counts).cuda(algorithm.gpu)
        algorithm.print_fn("unlabeled data class weights: {}".format(algorithm.x_ulb_weights))


class SAWCELoss(CELoss):
    def __init__(self):
        super().__init__()

    def set_weights(self, weights):
        self.x_lb_weights = weights

    def forward(self, logits, targets, reduction='none'):
        loss = super().forward(logits, targets, reduction='none')
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)
        loss = loss * self.x_lb_weights[targets]
        return loss.mean()


class SAWConsistencyLoss(ConsistencyLoss):
    def __init__(self):
        super().__init__()

    def set_weights(self, weights):
        self.x_ulb_weights = weights

    def forward(self, logits, targets, name='ce', mask=None):
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)
        if mask is None:
            mask = self.x_ulb_weights[targets]
        else:
            mask = mask * self.x_ulb_weights[targets]
        return super().forward(logits, targets, name, mask)