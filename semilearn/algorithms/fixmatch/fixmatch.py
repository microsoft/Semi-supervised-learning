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

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_lb, idx_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
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
            

            if(self.post_hoc_calib_conf is None):
                # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
                
                # if distribution alignment hook is registered, call it 
                # this is implemented for imbalanced algorithm - CReST
                if self.registered_hook("DistAlignHook"):
                    probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

                # compute mask
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

                # generate unlabeled targets using pseudo label hook
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)
                
                if(self.accumulate_pseudo_labels):
                    #new mask 

                    # the points that are newly psuedo-labeled
                    #mask2 = torch.logical_and(torch.logical_not(self.mask.ge(1.0)[idx_ulb]), mask) 

                    #self.print_fn(f"{torch.sum(mask).item()}, {torch.sum(mask2).item()}, {torch.sum(self.mask).item()}")
                    #print('here')
                    #print(mask)
                    idx_ulb = idx_ulb.to(self.device)

                    mask_bool = mask.ge(1.0)
                    #print(mask_bool)
                    

                    idx_ulb_pl = idx_ulb[mask_bool]
                    self.pseudo_labels[idx_ulb_pl] = pseudo_label[mask_bool]

                    self.mask[idx_ulb] = torch.clamp( mask + self.mask[idx_ulb], min=0.0, max=1.0) #torch.logical_or(mask_bool, self.mask[idx_ulb].ge(1.0)).to(self.mask[idx_ulb].dtype).to(self.device) 
                    
                    #print(idx_ulb, len(self.mask)) 
                    self.print_fn(f"{torch.sum(mask).item()},  {torch.sum( self.mask[idx_ulb] ).item()}, {torch.sum(self.mask).item()} ")
                    

                    pseudo_label = self.pseudo_labels[idx_ulb]
                    mask         = self.mask[idx_ulb]

                #else:
                    #self.pseudo_labels = y_hat 
                    #self.mask = scores.ge(tt).to(scores.dtype)
                    #pass

            else: 
                
                pseudo_label = self.pseudo_labels[idx_ulb]
                mask         = self.mask[idx_ulb]

            #print(pseudo_label, mask)
            batch_cov = (torch.sum(mask)/ len(idx_ulb)).item()
            self.print_fn(f"{batch_cov}")

            if not self.tb_log is None:
                self.tb_log.update({"batch_cov":batch_cov}, self.it)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]