# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch 
import torch.nn as nn
import numpy as np
from inspect import signature

from .utils import TRASLogitsAdjCELoss, TRASKLLoss
from semilearn.core import ImbAlgorithmBase
from semilearn.algorithms.utils import SSL_Argument
from semilearn.core.utils import IMB_ALGORITHMS


class TRASNet(nn.Module):
    """
        Transfer & Share algorithm (https://arxiv.org/abs/2205.13358).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - tras_A
                A parameter in TRAS
            - tras_B
                B parameter  in TRAS
            - tras_tro:
                tro parameter in TRAS
            - tras_warmup_epochs:
                TRAS warmup epochs
    """
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


@IMB_ALGORITHMS.register('tras')
class TRAS(ImbAlgorithmBase):
    def __init__(self, args, **kwargs):
        self.imb_init(A=args.tras_A, B=args.tras_B, tro=args.tras_tro, warmup_epochs=args.tras_warmup_epochs)
        super().__init__(args, **kwargs)
        assert args.algorithm == 'fixmatch', "Adsh only supports FixMatch as the base algorithm."

        # comput lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        
        # TODO: better ways
        self.model = TRASNet(self.model, num_classes=self.num_classes)
        self.ema_model = TRASNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

        # compute T logits
        self.la = torch.log(self.lb_class_dist ** self.tro + 1e-12).to(self.gpu)
        T_logit = torch.softmax(-self.la / 1, dim=0)
        self.T_logit = self.A * T_logit + self.B

        # create tras ce loss
        self.tras_ce_loss = TRASLogitsAdjCELoss(la=self.la)
        self.tras_kl_loss = TRASKLLoss()

    def imb_init(self, A, B, tro, warmup_epochs):
        self.A = A
        self.B = B
        self.tro = tro
        self.warmup_epochs = warmup_epochs

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):
        out_dict, log_dict = super().train_step(*args, **kwargs)

        if self.epoch < self.warmup_epochs:
            return out_dict, log_dict
        
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
        
        # compute supervised loss 
        tras_sup_loss = self.tras_ce_loss(logits_x_lb, kwargs['y_lb'], reduction='mean')

        # compute mask
        probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
        mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

        # generate unlabeled targets using pseudo label hook
        pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                        logits=probs_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T,
                                        softmax=False)

        la_u = self.la.expand([logits_x_ulb_s.size(0), self.num_classes])
        la_u = (la_u.t() * self.T_logit[pseudo_label].cuda()).t()

        # TRAS loss of unlabeled
        tras_unsup_loss = self.tras_kl_loss(logits_x_ulb_s, logits_x_ulb_w.detach()-la_u, 1, mask)

        tras_loss = tras_sup_loss + tras_unsup_loss

        out_dict['loss'] += tras_loss
        log_dict['train/tras_loss'] = tras_loss.item()
        return out_dict, log_dict

    def compute_prob(self, logits):
        return super().compute_prob(logits - self.la)

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        if self.epoch < self.warmup_epochs:
            out_key = 'logits'
        else:
            out_key = 'logits_aux'
        return super().evaluate(eval_dest=eval_dest, out_key=out_key, return_logits=return_logits)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--tras_warmup_epochs', int, 10),
            SSL_Argument('--tras_A', int, 2),
            SSL_Argument('--tras_B', int, 2),
            SSL_Argument('--tras_tro', float, 1.0),
        ]

        