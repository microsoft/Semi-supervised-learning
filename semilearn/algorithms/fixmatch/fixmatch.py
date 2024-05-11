# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

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
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
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

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
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

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            self.acc_pseudo_labels_flag = True 
            
            if(self.batch_pl_flag):
                
                # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
                
                # if distribution alignment hook is registered, call it 
                # this is implemented for imbalanced algorithm - CReST
                if self.registered_hook("DistAlignHook"):
                    probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

                # compute mask
                mask_batch = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

                # generate unlabeled targets using pseudo label hook
                pseudo_labels_batch = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)
                print(mask_batch.sum())

                if(self.acc_pseudo_labels_flag):
                    pseudo_labels_batch, mask_batch = self.accumulate_pseudo_labels(idx_ulb, mask_batch,
                                                                                     pseudo_labels_batch)

            else:
                # either global_pl_flag is True or post_hoc_calib is true. 
                pseudo_labels_batch = self.pseudo_labels[idx_ulb]
                mask_batch         = self.mask[idx_ulb]
            

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_labels_batch,
                                               'ce',
                                               mask=mask_batch)

            '''
            c = self.agg_pl_cov.detach().item()
            if(self.args.loss_reweight):
                # reweight loss by  
                # c will go high as the coverage increases. 
                # it reflects the portion of training data that is pseudo-labeled
                c = 0.5
                
                if(self.agg_pl_cov>0.8):
                    c = (self.n_a)/(self.n_l + self.n_a)

                total_loss = (1-c)*sup_loss + c * self.lambda_u* unsup_loss

                self.print_fn(f"loss weights : {1-c},  {c}") 

            else:
            '''
            
            total_loss = sup_loss + self.lambda_u * unsup_loss
        
        self.log_batch_pseudo_labeling_stats(mask_batch,pseudo_labels_batch,idx_ulb)
        self.log_full_pseudo_labeling_stats()

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask_batch.float().mean().item())
        return out_dict, log_dict


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]