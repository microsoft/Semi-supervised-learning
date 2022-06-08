
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.algorithms.algorithmbase import AlgorithmBase
from src.algorithms.utils import ce_loss, consistency_loss, EMA, SSL_Argument, str2bool
from src.datasets.samplers.sampler import DistributedSampler


class CoMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(CoMatch_Net, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_planes, proj_size)
        ])
        
    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return logits, feat_proj 


def comatch_contrastive_loss(feats_x_ulb_s_0, feats_x_ulb_s_1, Q, T=0.2):
    # embedding similarity
    sim = torch.exp(torch.mm(feats_x_ulb_s_0, feats_x_ulb_s_1.t())/ T) 
    sim_probs = sim / sim.sum(1, keepdim=True)
    # contrastive loss
    loss = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
    loss = loss.mean()  
    return loss


class CoMatch(AlgorithmBase):
    """
        CoMatch algorithm (https://arxiv.org/abs/2011.11183).
        Reference implementation (https://github.com/salesforce/CoMatch/).

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
            - contrast_p_cutoff (`float`):
                Confidence threshold for contrastive loss. Samples with similarity lower than a threshold are not connected.
            - queue_batch (`int`, *optional*, default to 128):
                Length of the memory bank to store class probabilities and embeddings of the past weakly augmented samples
            - smoothing_alpha (`float`, *optional*, default to 0.999):
                Weight for a smoothness constraint which encourages taking a similar value as its nearby samples’ class probabilities
            - da_len (`int`, *optional*, default to 256):
                Length of the memory bank for distribution alignment.
            - contrast_loss_ratio (`float`, *optional*, default to 1.0):
                Loss weight for contrastive loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # comatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, 
                  contrast_p_cutoff=args.contrast_p_cutoff, hard_label=args.hard_label, 
                  queue_batch=args.queue_batch, smoothing_alpha=args.smoothing_alpha, da_len=args.da_len)
        self.lambda_c = args.contrast_loss_ratio
        # warp model
        backbone = self.model
        self.model = CoMatch_Net(backbone, proj_size=self.args.proj_size)
        self.ema_model = CoMatch_Net(self.ema_model, proj_size=self.args.proj_size)
        self.ema_model.load_state_dict(self.model.state_dict())

    def init(self, T, p_cutoff, contrast_p_cutoff, hard_label=True, queue_batch=128, smoothing_alpha=0.999, da_len=256):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.contrast_p_cutoff = contrast_p_cutoff
        self.use_hard_label = hard_label
        self.queue_batch = queue_batch
        self.smoothing_alpha = smoothing_alpha
        self.da_len = da_len

        # memory smoothing
        self.queue_size = int(queue_batch * (self.args.uratio + 1) * self.args.batch_size)
        self.queue_feats = torch.zeros(self.queue_size, self.args.proj_size).cuda(self.gpu)
        self.queue_probs = torch.zeros(self.queue_size, self.args.num_classes).cuda(self.gpu)
        self.queue_ptr = 0

        # distribution alignment
        self.da_len = da_len
        if self.da_len:
            self.da_queue = torch.zeros(self.da_len, self.num_classes, dtype=torch.float).cuda(self.gpu)
            self.da_ptr = torch.zeros(1, dtype=torch.long).cuda(self.gpu)

    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.mean(0)
        ptr = int(self.da_ptr)

        if self.distributed:
            torch.distributed.all_reduce(probs_bt_mean)
            self.da_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()
        else:
            self.da_queue[ptr] = probs_bt_mean

        self.da_ptr[0] = (ptr + 1) % self.da_len
        probs = probs / self.da_queue.mean(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()
    
    # utils
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

    @torch.no_grad()
    def update_bank(self, feats, probs):
        if self.distributed:
            feats = self.concat_all_gather(feats)
            probs = self.concat_all_gather(probs)
        # update memory bank
        length = feats.shape[0]
        self.queue_feats[self.queue_ptr:self.queue_ptr + length, :] = feats
        self.queue_probs[self.queue_ptr:self.queue_ptr + length, :] = probs      
        self.queue_ptr = (self.queue_ptr + length) % self.queue_size


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb = y_lb.shape[0] 

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                logits, feats = self.model(inputs)
                logits_x_lb, feats_x_lb = logits[:num_lb], feats[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s_0, _ = logits[num_lb:].chunk(3)
                feats_x_ulb_w, feats_x_ulb_s_0, feats_x_ulb_s_1 = feats[num_lb:].chunk(3)
            else:            
                logits_x_lb, feats_x_lb = self.model(x_lb)
                logits_x_ulb_s_0, feats_x_ulb_s_0 = self.model(x_ulb_s_0)
                _, feats_x_ulb_s_1 = self.model(x_ulb_s_1)
                with torch.no_grad():
                    logits_x_ulb_w, feats_x_ulb_w = self.model(x_ulb_w)

            # supervised loss
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            
            with torch.no_grad():
                logits_x_ulb_w = logits_x_ulb_w.detach()
                feats_x_lb = feats_x_lb.detach()
                feats_x_ulb_w = feats_x_ulb_w.detach()

                probs = torch.softmax(logits_x_ulb_w, dim=1)            
                # distribution alignment
                if self.da_len:
                    probs = self.distribution_alignment(probs)

                probs_orig = probs.clone()
                # memory-smoothing 
                if self.epoch >0 and self.it > self.queue_batch: 
                    A = torch.exp(torch.mm(feats_x_ulb_w, self.queue_feats.t()) / self.T)       
                    A = A / A.sum(1,keepdim=True)                    
                    probs = self.smoothing_alpha * probs + (1 - self.smoothing_alpha) * torch.mm(A, self.queue_probs)    
                
                max_probs, _ = torch.max(probs, dim=1)
                mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)
                    
                feats_w = torch.cat([feats_x_ulb_w, feats_x_lb],dim=0)   
                probs_w = torch.cat([probs_orig, F.one_hot(y_lb, num_classes=self.num_classes)],dim=0)

                self.update_bank(feats_w, probs_w)


            unsup_loss, _ = consistency_loss(logits_x_ulb_s_0,
                                             probs,
                                             'ce',
                                             use_hard_labels=False,
                                             T=1.0,
                                             mask=mask,
                                             softmax=False)

            # pseudo-label graph with self-loop
            Q = torch.mm(probs, probs.t())       
            Q.fill_diagonal_(1)    
            pos_mask = (Q >= self.contrast_p_cutoff).to(mask.dtype)
            Q = Q * pos_mask
            Q = Q / Q.sum(1, keepdim=True)

            contrast_loss = comatch_contrastive_loss(feats_x_ulb_s_0, feats_x_ulb_s_1, Q, T=self.T)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * contrast_loss

        # parameter updates
        self.parameter_update(total_loss)


        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/contrast_loss'] = contrast_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict =  super().get_save_dict()
        save_dict['queue_feats'] = self.queue_feats.cpu()
        save_dict['queue_probs'] = self.queue_probs.cpu()
        save_dict['queue_size'] = self.queue_size
        save_dict['queue_ptr'] = self.queue_ptr
        if self.da_len:
            save_dict['da_queue'] = self.da_queue.cpu()
            save_dict['da_ptr'] = self.da_ptr.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.queue_feats = checkpoint['queue_feats'].cuda(self.gpu)
        self.queue_probs = checkpoint['queue_probs'].cuda(self.gpu)
        self.queue_size = checkpoint['queue_size']
        self.queue_ptr = checkpoint['queue_ptr']
        if self.da_len:
            self.da_queue = checkpoint['da_queue'].cuda(self.gpu)
            self.da_ptr = checkpoint['da_ptr'].cuda(self.gpu)
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--contrast_p_cutoff', float, 0.8),
            SSL_Argument('--contrast_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--queue_batch', int, 128),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
        ]
