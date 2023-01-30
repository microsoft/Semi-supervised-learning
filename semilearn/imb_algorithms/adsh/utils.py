import torch
import numpy as np
from semilearn.algorithms.hooks import MaskingHook


class AdaptiveThresholdingHook(MaskingHook):
    def __init__(self, num_classes, tau_1):
        super(AdaptiveThresholdingHook, self).__init__()
        self.adsh_s = torch.ones((num_classes,)) * tau_1

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if softmax_x_ulb:
            # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            probs_x_ulb = self.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, pred = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(torch.exp(-self.adsh_s.to(max_probs.device)[pred])).to(max_probs.dtype)
        return mask

    def after_train_epoch(self, algorithm):

        logits_x_ulb = algorithm.evaluate('eval_ulb', return_logits=True)['eval_ulb/logits']

        logits_x_ulb = torch.from_numpy(logits_x_ulb)
        # p_x_ulb_w = torch.softmax(logits_x_ulb, dim=-1)
        p_x_ulb_w = algorithm.compute_prob(logits_x_ulb)
        conf_all, pred_all = torch.max(p_x_ulb_w, dim=-1)

        C = []
        for y in range(algorithm.num_classes):
            C.append(torch.sort(conf_all[pred_all == y], descending=True)[0])  # descending order

        rho = 1.0
        for i in range(len(C[0])):
            if C[0][i] < algorithm.tau_1:
                break
            rho = i / len(C[0])

        for k in range(algorithm.num_classes):
            if len(C[k]) != 0:
                self.adsh_s[k] = - torch.log(C[k][int(len(C[k]) * rho) - 1])
            # todo: how to update s when len(C[k]) == 0
