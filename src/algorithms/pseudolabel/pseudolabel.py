# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from src.algorithms.algorithmbase import AlgorithmBase
from src.algorithms.utils import ce_loss, consistency_loss, EMA, SSL_Argument
from src.datasets.samplers.sampler import DistributedSampler


class PseudoLabel(AlgorithmBase):
    """
        Pseudo Label algorithm (https://arxiv.org/abs/1908.02983).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(p_cutoff=args.p_cutoff, unsup_warm_up=args.unsup_warm_up)

    def init(self, p_cutoff, unsup_warm_up=0.4):
        self.p_cutoff = p_cutoff
        self.unsup_warm_up = unsup_warm_up 

    def train_step(self, x_lb, y_lb, x_ulb_w):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)
            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            logits_x_ulb = self.model(x_ulb_w)
            self.bn_controller.unfreeze_bn(self.model)


            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # compute mask
            with torch.no_grad():
                max_probs = torch.max(torch.softmax(logits_x_ulb.detach(), dim=-1), dim=-1)[0]
                mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)

            unsup_loss, _ = consistency_loss(logits_x_ulb,
                                             logits_x_ulb,
                                             'ce',
                                             mask=mask)

            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup

        # parameter updates
        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
            # SSL_Argument('--use_flex', str2bool, False),
        ]