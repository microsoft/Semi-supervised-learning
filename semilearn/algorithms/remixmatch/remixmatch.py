# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignEMAHook 
from semilearn.algorithms.utils import str2bool, mixup_one_target, SSL_Argument


class ReMixMatch_Net(nn.Module):
    def __init__(self, base, use_rot=True):
        super(ReMixMatch_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features

        if use_rot:
            self.rot_classifier = nn.Linear(self.num_features, 4)

    def forward(self, x, use_rot=False, **kwargs):
        if not use_rot:
            return self.backbone(x, **kwargs)
        
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        logits_rot = self.rot_classifier(feat)
        return {'logits':logits, 'logits_rot':logits_rot, 'feat':feat}
    
    def init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()
    
    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('remixmatch')
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
        # remixmatch specified arguments
        self.lambda_rot = args.rot_loss_ratio
        self.lambda_kl = args.kl_loss_ratio
        self.use_rot = self.lambda_rot > 0
        super().__init__(args, net_builder,  tb_log, logger)
        self.init(T=args.T, unsup_warm_up=args.unsup_warm_up, mixup_alpha=args.mixup_alpha, mixup_manifold=args.mixup_manifold)

    def init(self, T, unsup_warm_up=0.4, mixup_alpha=0.5, mixup_manifold=False):
        self.T = T
        self.unsup_warm_up = unsup_warm_up
        self.mixup_alpha = mixup_alpha
        self.mixup_manifold = mixup_manifold

    def set_hooks(self):
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        lb_class_dist = lb_class_dist / lb_class_dist.sum()
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, p_target_type='gt', p_target=lb_class_dist), 
            "DistAlignHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = ReMixMatch_Net(model, self.use_rot)
        return model
    
    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = ReMixMatch_Net(ema_model, self.use_rot)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, x_ulb_s_0_rot=None, rot_v=None):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                # logits_x_lb = self.model(x_lb)[0]

                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
                # logits_x_ulb_w = self.model(x_ulb_w)
                # logits_x_ulb_s1 = self.model(x_ulb_s1)[0]
                # logits_x_ulb_s2 = self.model(x_ulb_s2)[0]
                self.bn_controller.unfreeze_bn(self.model)

                prob_x_ulb = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=self.compute_prob(logits_x_ulb_w))
                sharpen_prob_x_ulb = prob_x_ulb ** (1 / self.T)
                sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

            
            self.bn_controller.freeze_bn(self.model)
            outs_x_lb = self.model(x_lb)
            outs_x_ulb_s_0 = self.model(x_ulb_s_0)
            outs_x_ulb_s_1 = self.model(x_ulb_s_1)
            self.bn_controller.unfreeze_bn(self.model)

            feat_dict = {'x_lb':outs_x_lb['feat'], 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':[outs_x_ulb_s_0['feat'], outs_x_ulb_s_1['feat']]}

            # mix up
            # with torch.no_grad():
            input_labels = torch.cat([F.one_hot(y_lb, self.num_classes), sharpen_prob_x_ulb, sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)
            if self.mixup_manifold:
                inputs = torch.cat([outs_x_lb['feat'], outs_x_ulb_s_0['feat'], outs_x_ulb_s_1['feat'],  outs_x_ulb_w['feat']], dim=0)
            else:
                inputs = torch.cat([x_lb, x_ulb_s_0, x_ulb_s_1, x_ulb_w])
            mixed_x, mixed_y, _ = mixup_one_target(inputs, input_labels, self.mixup_alpha, is_bias=True)
            mixed_x = list(torch.split(mixed_x, num_lb))
            # mixed_x = interleave(mixed_x, num_lb)

            # calculate BN only for the first batch
            if self.mixup_manifold:
                logits = [self.model(mixed_x[0], only_fc=self.mixup_manifold)]
                # calculate BN for only the first batch
                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt, only_fc=self.mixup_manifold))
                self.bn_controller.unfreeze_bn(self.model)
            else:
                logits = [self.model(mixed_x[0])['logits']]
                # calculate BN for only the first batch
                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt)['logits'])
                self.bn_controller.unfreeze_bn(self.model)
            u1_logits = outs_x_ulb_s_0['logits']

            # put interleaved samples back
            # logits = interleave(logits, num_lb)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            # sup loss
            sup_loss = self.ce_loss(logits_x, mixed_y[:num_lb], reduction='mean')
            
            # unsup_loss
            unsup_loss = self.consistency_loss(logits_u, mixed_y[num_lb:])
            
            # loss U1
            u1_loss = self.consistency_loss(u1_logits, sharpen_prob_x_ulb)

            # ramp for w_match
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_kl * unsup_warmup * u1_loss + self.lambda_u * unsup_warmup * unsup_loss

            # calculate rot loss with w_rot
            if self.use_rot:
                self.bn_controller.freeze_bn(self.model)
                logits_rot = self.model(x_ulb_s_0_rot, use_rot=True)['logits_rot']
                self.bn_controller.unfreeze_bn(self.model)
                rot_loss = F.cross_entropy(logits_rot, rot_v, reduction='mean')
                rot_loss = rot_loss.mean()
                total_loss += self.lambda_rot * rot_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item())
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict
    
    def load_model(self, load_path):
        checkpoint =  super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
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
