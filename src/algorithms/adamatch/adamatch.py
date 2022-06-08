# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from src.algorithms.algorithmbase import AlgorithmBase
from src.algorithms.utils import ce_loss, consistency_loss,  SSL_Argument, str2bool


class AdaMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, ema_p=args.ema_p)
    
    def init(self, p_cutoff, T, hard_label=True, dist_align=True, ema_p=0.999):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.ema_p = ema_p
        
        self.lb_prob_t = torch.ones((self.args.num_classes)).cuda(self.args.gpu) / self.args.num_classes
        self.ulb_prob_t = torch.ones((self.args.num_classes)).cuda(self.args.gpu) / self.args.num_classes
    
    @torch.no_grad()
    def update_prob_t(self, lb_probs, ulb_probs):
        if self.args.distributed and self.args.world_size > 1:
            lb_probs = self.concat_all_gather(lb_probs)
            ulb_probs = self.concat_all_gather(ulb_probs)
        
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

        lb_prob_t = lb_probs.mean(0)
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t


    @torch.no_grad()
    def distribution_alignment(self, probs):
        # da
        probs = probs * (1e-6 + self.lb_prob_t) / (1e-6 + self.ulb_prob_t)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
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

            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # update 
            self.update_prob_t(probs_x_lb, probs_x_ulb_w)

            # distribution alignment
            if self.dist_align:
                probs_x_ulb_w = self.distribution_alignment(probs_x_ulb_w)

            # calculate weight
            max_probs, _ = probs_x_lb.max(dim=-1)
            p_cutoff = max_probs.mean() * self.p_cutoff

            max_probs, max_idx = probs_x_ulb_w.max(dim=-1)
            mask = max_probs.ge(p_cutoff).to(max_probs.dtype)
            # max_probs, mask = self.calculate_mask(probs_x_ulb_w)

            # calculate loss 
            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             probs_x_ulb_w,
                                             'ce',
                                             use_hard_labels=self.use_hard_label,
                                             T=self.T,
                                             mask=mask,
                                             softmax=False)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/avg_w'] = mask.mean().item()
        tb_dict['train/avg_max_prob'] = max_probs.mean().item()
        tb_dict['train/avg_prob_t'] = self.ulb_prob_t.mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['ulb_prob_t'] = self.ulb_prob_t.cpu()
        save_dict['lb_prob_t'] = self.lb_prob_t.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.ulb_prob_t = checkpoint['ulb_prob_t'].cuda(self.args.gpu)
        self.lb_prob_t = checkpoint['lb_prob_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]