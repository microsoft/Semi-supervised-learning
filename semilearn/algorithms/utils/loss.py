# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F


def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist


def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

    # if use_hard_labels:
    #     log_pred = F.log_softmax(logits, dim=-1)
    #     return F.nll_loss(log_pred, targets, reduction=reduction)
    #     # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    # else:
    #     assert logits.shape == targets.shape
    #     log_pred = F.log_softmax(logits, dim=-1)
    #     nll_loss = torch.sum(-targets * log_pred, dim=1)
    #     if reduction == 'none':
    #         return nll_loss
    #     else:
    #         return nll_loss.mean()


# TODO: need to split this function into pseudo label and computing loss 
def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits_s: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        logits_w: logit to provide pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        use_hard_labels: flag of using hard pseudo labels
        T: temperature for shapring the probability
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
        label_smoothing: label smoothing value when calculate cross-entropy loss
        softmax: flag of whether logits_w need to go through softmax function
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        # if softmax:
        #     pseudo_label = torch.softmax(logits_w, dim=-1)
        # else:
        #     pseudo_label = logits_w
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        # if use_hard_labels:
        # pseudo_label = torch.argmax(logits_w, dim=-1)
        # if label_smoothing:
        #     pseudo_label = smooth_targets(logits_w, pseudo_label, label_smoothing)
        #     use_hard_labels = False
        loss = ce_loss(logits, targets, reduction='none')
        # else:
        #     if softmax:
        #         pseudo_label = torch.softmax(logits_w / T, dim=-1)
        #     else:
        #         pseudo_label = logits_w
        #     loss = ce_loss(logits_s, pseudo_label, use_hard_labels, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()