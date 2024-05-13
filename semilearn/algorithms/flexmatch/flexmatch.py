
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import FlexMatchThresholdingHook

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('flexmatch')
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
        # flexmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
    
    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_lb, idx_ulb, x_ulb):

        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses

        # aug_1 : data augmentation to get pseudolabels "id" or "weak", "strong"
        # aug_1 is weak for vanilla fixmatch

        # aug_2:  data augmentation in consistency loss "id" or "weak", "strong"
        # aug_2 is strong for vanilla fixmatch

        if(self.aug_1=="id" and self.aug_2=="id"):
            x_ulb_w = x_ulb
            x_ulb_s = x_ulb 

        elif(self.aug_1=="weak" and self.aug_2=="weak"):
            x_ulb_s =  x_ulb_w 

        elif(self.aug_1=="id" and self.aug_2 =="weak"):
            x_ulb_w = x_ulb 
            x_ulb_s = x_ulb_w 

        elif(self.aug_1=="weak" and self.aug_2=="strong"):
            pass 

        else:
            self.print_fn('XXXXXXXX Combination not supported XXXXXXXX') 
            # strong, strong 
            # strong id
            # id     strong
            # strong weak 
            # weak   id 

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                rep, logits, Lkl, logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, feats_x_lb, feats_x_ulb_w, feats_x_ulb_s = self.use_cat_func(x_lb, x_ulb_w, x_ulb_s, num_lb)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            self.sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            # if using BaM, the function run bam related stuff
            self.check_if_use_bam(Lkl, logits, num_lb, rep)

            if(self.batch_pl_flag):
                # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

                # if distribution alignment hook is registered, call it 
                # this is implemented for imbalanced algorithm - CReST
                
                if self.registered_hook("DistAlignHook"):
                    probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

                # compute mask
                mask_batch = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
                
                # generate unlabeled targets using pseudo label hook
                pseudo_labels_batch = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)
            else:
                # either global_pl_flag is True or post_hoc_calib is true. 
                pseudo_labels_batch = self.pseudo_labels[idx_ulb]
                mask_batch          = self.mask[idx_ulb]

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_labels_batch,
                                               'ce',
                                               mask=mask_batch)

            total_loss = self.sup_loss + self.lambda_u * unsup_loss
        
        self.log_batch_pseudo_labeling_stats(mask_batch,pseudo_labels_batch,idx_ulb)
        self.log_full_pseudo_labeling_stats()

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=self.sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask_batch.float().mean().item())
        return out_dict, log_dict
        
    

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
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