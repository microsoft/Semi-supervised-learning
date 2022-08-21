# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.algorithms.algorithmbase import AlgorithmBase
from semilearn.algorithms.utils import ce_loss, SSL_Argument, str2bool, interleave



class ReMixMatch_Net(nn.Module):
    def __init__(self, base, use_rot=True):
        super(ReMixMatch_Net, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features

        if use_rot:
            self.rot_classifier = nn.Linear(self.feat_planes, 4)

    def forward(self, x, use_rot=False, **kwargs):
        if not use_rot:
            return self.backbone(x, **kwargs)
        
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        logits_rot = self.rot_classifier(feat)
        return logits, logits_rot


class ReMixMatch(AlgorithmBase):
    """
    ReMixMatch algorithm (https://arxiv.org/abs/1911.09785).

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
        - rot_loss_ratio ('float',  *optional*, defaults to 0.5):
            rotation loss weight
        - kl_loss_ratio ('float',  *optional*, defaults to 0.5):
            kl loss weight
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder,  tb_log, logger)
        # remixmatch specificed arguments
        self.init(T=args.T, unsup_warm_up=args.unsup_warm_up, mixup_alpha=args.mixup_alpha, mixup_manifold=args.mixup_manifold)
        self.lambda_rot = args.rot_loss_ratio
        self.lambda_kl = args.kl_loss_ratio
        self.use_rot = self.lambda_rot > 0
        self.model = ReMixMatch_Net(self.model, self.use_rot)
        self.ema_model = ReMixMatch_Net(self.ema_model, self.use_rot)
        self.ema_model.load_state_dict(self.model.state_dict())

    def init(self, T, unsup_warm_up=0.4, mixup_alpha=0.5, mixup_manifold=False):
        self.T = T
        self.unsup_warm_up = unsup_warm_up
        self.mixup_alpha = mixup_alpha
        self.mixup_manifold = mixup_manifold

        # p(y) based on the labeled examples seen during training
        try:
            dist_file_name = r"./data_statistics/" + self.args.dataset + '_' + str(self.args.num_labels) + '.json'
            with open(dist_file_name, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
                self.p_target = p_target.cuda(self.gpu)
            print('p_target:', self.p_target)
        except:
            self.p_target = torch.ones((self.num_classes, )).to(self.gpu) / self.num_classes
        self.p_model = None


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, x_ulb_s_0_rot=None, rot_v=None):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                # logits_x_lb = self.model(x_lb)[0]
                logits_x_ulb_w = self.model(x_ulb_w)
                # logits_x_ulb_s1 = self.model(x_ulb_s1)[0]
                # logits_x_ulb_s2 = self.model(x_ulb_s2)[0]
                self.bn_controller.unfreeze_bn(self.model)

                prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)

                # p^~_(y): moving average of p(y)
                # TODO: add distribution alignment to utils
                if self.p_model == None:
                    self.p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                else:
                    self.p_model = self.p_model * 0.999 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.001

                prob_x_ulb = prob_x_ulb * self.p_target / self.p_model
                prob_x_ulb = (prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True))

                sharpen_prob_x_ulb = prob_x_ulb ** (1 / self.T)
                sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

            # mix up
            # with torch.no_grad():
            input_labels = torch.cat([F.one_hot(y_lb, self.num_classes), sharpen_prob_x_ulb, sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)
            if self.mixup_manifold:
                inputs = torch.cat([self.model(x_lb, only_feat=True), self.model(x_ulb_s_0, only_feat=True), self.model(x_ulb_s_1, only_feat=True), self.model(x_ulb_w, only_feat=True)])
            else:
                inputs = torch.cat([x_lb, x_ulb_s_0, x_ulb_s_1, x_ulb_w])
            mixed_x, mixed_y, _ = self.mixup_one_target(inputs, input_labels, self.mixup_alpha, is_bias=True)
            mixed_x = list(torch.split(mixed_x, num_lb))
            mixed_x = interleave(mixed_x, num_lb)

            # calculate BN only for the first batch
            logits = [self.model(mixed_x[0], only_fc=self.mixup_manifold)]
            self.bn_controller.freeze_bn(self.model)
            for ipt in mixed_x[1:]:
                logits.append(self.model(ipt, only_fc=self.mixup_manifold))
            u1_logits = self.model(x_ulb_s_0)
            self.bn_controller.unfreeze_bn(self.model)

            # put interleaved samples back
            logits = interleave(logits, num_lb)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            # sup loss
            sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False, reduction='mean')
            
            # unsup_loss
            unsup_loss = ce_loss(logits_u, mixed_y[num_lb:], use_hard_labels=False,  reduction='mean')
            
            # loss U1
            u1_loss = ce_loss(u1_logits, sharpen_prob_x_ulb, use_hard_labels=False,  reduction='mean')

            # ramp for w_match
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_kl * unsup_warmup * u1_loss + self.lambda_u * unsup_warmup * unsup_loss

            # calculate rot loss with w_rot
            if self.use_rot:
                self.bn_controller.freeze_bn(self.model)
                logits_rot = self.model(x_ulb_s_0_rot, use_rot=True)[1]
                self.bn_controller.unfreeze_bn(self.model)
                rot_loss = ce_loss(logits_rot, rot_v, reduction='mean')
                rot_loss = rot_loss.mean()
                total_loss += self.lambda_rot * rot_loss

        # parameter updates
        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()

        return tb_dict

    
    # TODO: move mixup to utils
    @torch.no_grad()
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

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['p_model'] = self.p_model.cpu()
        save_dict['p_target'] = self.p_target.cpu()
        return save_dict
    
    def load_model(self, load_path):
        checkpoint =  super().load_model(load_path)
        self.p_model = checkpoint['p_model'].cuda(self.gpu)
        self.p_target = checkpoint['p_target'].cuda(self.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5, 'Temperature Sharpening'),
            SSL_Argument('--kl_loss_ratio', float, 0.5, 'weight for KL loss'),
            SSL_Argument('--rot_loss_ratio', float, 0.5, 'weight for rot loss, set to 0 for nlp and speech'),
            SSL_Argument('--unsup_warm_up', float, 1 / 64),
            SSL_Argument('--mixup_alpha', float, 0.75, 'param for Beta distribution of Mix Up'),
            SSL_Argument('--mixup_manifold', str2bool, False, 'use manifold mixup (for nlp)'),
        ]
