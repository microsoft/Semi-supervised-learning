
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
# from copy import deepcopy
# from collections import Counter

from .utils import FlexMatchThresholdingHook

from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool



class FlexMatch(AlgorithmBase):
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).

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
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
    
    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

        # # how to init this
        # self.ulb_dest_len = ulb_dest_len
        # selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1
        # self.selected_label = selected_label.cuda(self.gpu)
        # self.classwise_acc = torch.zeros((self.num_classes,)).cuda(self.gpu)

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        super().set_hooks()

    # @torch.no_grad()
    # def update_classwise_acc(self):
    #     pseudo_counter = Counter(self.selected_label.tolist())
    #     if max(pseudo_counter.values()) < self.ulb_dest_len:  # not all(5w) -1
    #         if self.thresh_warmup:
    #             for i in range(self.num_classes):
    #                 self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
    #         else:
    #             wo_negative_one = deepcopy(pseudo_counter)
    #             if -1 in wo_negative_one.keys():
    #                 wo_negative_one.pop(-1)
    #             for i in range(self.num_classes):
    #                 self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                logits = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            else:
                logits_x_lb = self.model(x_lb)
                logits_x_ulb_s = self.model(x_ulb_s)
                with torch.no_grad():
                    logits_x_ulb_w = self.model(x_ulb_w)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # compute mask
            # TODO: move this into masking hook
            # with torch.no_grad():
            #     max_probs, max_idx = torch.max(torch.softmax(logits_x_ulb_w.detach(), dim=-1), dim=-1)
            #     # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
            #     # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
            #     mask = max_probs.ge(self.p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))  # convex
            #     # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
            #     select = max_probs.ge(self.p_cutoff)
            #     mask = mask.to(max_probs.dtype)
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w, idx_ulb=idx_ulb)
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

            # this is in hook
            # # update classwise acc
            # if idx_ulb[select == 1].nelement() != 0:
            #     self.selected_label[idx_ulb[select == 1]] = pseudo_label[select == 1]
            # self.update_classwise_acc()

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['selected_label'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['classwise_acc'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]
