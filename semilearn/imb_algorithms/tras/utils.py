# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core.criterions import CELoss


class TRASLogitsAdjCELoss(CELoss):
    def __init__(self, la):
        super().__init__()
        self.la = la

    def forward(self, logits, targets, reduction='mean'):
        return super().forward(logits + self.la, targets, reduction=reduction)

class TRASKLLoss(nn.Module):
    def forward(self, outputs, targets, T, mask):
        _p = F.log_softmax(outputs / T, dim=1)
        _q = F.softmax(targets / (T * 2), dim=1)
        _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1) * mask)
        _soft_loss = _soft_loss * T * T
        return _soft_loss
