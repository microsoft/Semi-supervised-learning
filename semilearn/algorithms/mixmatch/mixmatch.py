# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn.functional as F
from semilearn.algorithms.algorithmbase import AlgorithmBase
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool, interleave


class MixMatch(AlgorithmBase):
    """
        MixMatch algorithm (https://arxiv.org/abs/1905.02249).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - unsup_warm_up (`float`, *optional*, defaults to 0.4):
                Ramp up for weights for unsupervised loss
            - mixup_alpha (`float`, *optional*, defaults to 0.5):
                Hyper-parameter of mixup
            - mixup_manifold (`bool`, *optional*, defaults to `False`):
                Whether or not to use manifold mixup
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # mixmatch specificed arguments
        self.init(T=args.T, unsup_warm_up=args.unsup_warm_up, mixup_alpha=args.mixup_alpha, mixup_manifold=args.mixup_manifold)

    def init(self, T, unsup_warm_up=0.01525, mixup_alpha=0.5, mixup_manifold=False):
        self.T = T
        self.unsup_warm_up = unsup_warm_up
        self.mixup_alpha = mixup_alpha
        self.mixup_manifold = mixup_manifold

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                logits_x_ulb_w1 = self.model(x_ulb_w)
                logits_x_ulb_w2 = self.model(x_ulb_s)
                self.bn_controller.unfreeze_bn(self.model)
                
                # avg
                avg_prob_x_ulb = (torch.softmax(logits_x_ulb_w1, dim=1) + torch.softmax(logits_x_ulb_w2, dim=1)) / 2
                # avg_prob_x_ulb = (avg_prob_x_ulb / avg_prob_x_ulb.sum(dim=-1, keepdim=True))
                # sharpening
                sharpen_prob_x_ulb = avg_prob_x_ulb ** (1 / self.T)
                sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

            # with torch.no_grad():
            # Pseudo Label
            input_labels = torch.cat([F.one_hot(y_lb, self.num_classes), sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)
            # Mix up
            if self.mixup_manifold:
                inputs = torch.cat((self.model(x_lb, only_feat=True), self.model(x_ulb_w, only_feat=True), self.model(x_ulb_s, only_feat=True)))
            else:
                inputs = torch.cat([x_lb, x_ulb_w, x_ulb_s])
            mixed_x, mixed_y, _ = self.mixup_one_target(inputs, input_labels,
                                                        self.mixup_alpha,
                                                        is_bias=True)
            mixed_x = list(torch.split(mixed_x, num_lb))
            mixed_x = interleave(mixed_x, num_lb)

            logits = [self.model(mixed_x[0], only_fc=self.mixup_manifold)]
            # calculate BN for only the first batch
            self.bn_controller.freeze_bn(self.model)
            for ipt in mixed_x[1:]:
                logits.append(self.model(ipt, only_fc=self.mixup_manifold))
            self.bn_controller.unfreeze_bn(self.model)

            # put interleaved samples back
            logits = interleave(logits, num_lb)

            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False, reduction='mean')
            unsup_loss, _ = consistency_loss(logits_u, mixed_y[num_lb:], name='mse', softmax=False)

            # set ramp_up for lambda_u
            unsup_warmup = float(np.clip(self.it / (self.unsup_warm_up * self.num_train_iter), 0.0, 1.0))
            lambda_u = self.lambda_u * unsup_warmup

            total_loss = sup_loss + lambda_u * unsup_loss

        # parameter updates
        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        return tb_dict

    # TODO: move mixup to utils
    def mixup_one_target(self, x, y, alpha=1.0, is_bias=False):
        """Returns mixed inputs, mixed targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)

        index = torch.randperm(x.size(0)).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y, lam

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5, 'parameter for Temperature Sharpening'),
            SSL_Argument('--unsup_warm_up', float, 1 / 64, 'ramp up ratio for unsupervised loss'),
            SSL_Argument('--mixup_alpha', float, 0.5, 'parameter for Beta distribution of Mix Up'),
            SSL_Argument('--mixup_manifold', str2bool, False, 'use manifold mixup (for nlp)'),
        ]
