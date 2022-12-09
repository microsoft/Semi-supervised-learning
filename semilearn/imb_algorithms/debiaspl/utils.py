# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.core.hooks import Hook
from semilearn.core.criterions import ConsistencyLoss, consistency_loss

class DebiasPLHook(Hook):
    def before_train_step(self, algorithm):
        algorithm.consistency_loss.set_param(algorithm.p_hat)


class DebiasPLConsistencyLoss(ConsistencyLoss):
    def __init__(self, tau=0.4):
        super().__init__()
        self.tau = 0.4

    def set_param(self, p):
        self.p_hat = p

    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits + self.tau * torch.log(self.p_hat), targets, name, mask)