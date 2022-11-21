# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
from semilearn.algorithms.hooks import MaskingHook

class AdaMatchThresholdingHook(MaskingHook):
    """
    Relative Confidence Thresholding in AdaMatch
    """

    @torch.no_grad()
    def masking(self, algorithm, logits_x_lb, logits_x_ulb, softmax_x_lb=True, softmax_x_ulb=True,  *args, **kwargs):
        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        if softmax_x_lb:
            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_lb = algorithm.compute_prob(logits_x_lb.detach())
        else:
            # logits is already probs
            probs_x_lb = logits_x_lb.detach()

        max_probs, _ = probs_x_lb.max(dim=-1)
        p_cutoff = max_probs.mean() * algorithm.p_cutoff
        max_probs, _ = probs_x_ulb.max(dim=-1)
        mask = max_probs.ge(p_cutoff).to(max_probs.dtype)
        return mask