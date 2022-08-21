# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.algorithms.algorithmbase import AlgorithmBase
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool


class SimMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(SimMatch_Net, self).__init__()
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



class SimMatch(AlgorithmBase):
    """
    SimMatch algorithm (https://arxiv.org/abs/2203.06915).
    Reference implementation (https://github.com/KyleZheng1997/simmatch).

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
        - K (`int`, *optional*, default to 128):
            Length of the memory bank to store class probabilities and embeddings of the past weakly augmented samples
        - smoothing_alpha (`float`, *optional*, default to 0.999):
            Weight for a smoothness constraint which encourages taking a similar value as its nearby samples’ class probabilities
        - da_len (`int`, *optional*, default to 256):
            Length of the memory bank for distribution alignment.
        - in_loss_ratio (`float`, *optional*, default to 1.0):
            Loss weight for simmatch feature loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # simmatch specificed arguments
        # adjust k 
        self.use_ema_teacher = True
        if args.dataset in ['cifar10', 'cifar100', 'svhn', 'superks', 'tissuemnist', 'eurosat', 'superbks', 'esc50', 'gtzan', 'urbansound8k', 'aclImdb', 'ag_news', 'dbpedia']:
            self.use_ema_teacher = False
            self.ema_bank = 0.7
        args.K = args.lb_dest_len
        self.init(T=args.T, p_cutoff=args.p_cutoff, proj_size=args.proj_size, K=args.K, smoothing_alpha=args.smoothing_alpha, da_len=args.da_len)
        self.lambda_in = args.in_loss_ratio
    
        # warp model
        backbone = self.model
        self.model = SimMatch_Net(backbone, proj_size=self.args.proj_size)
        self.ema_model = SimMatch_Net(self.ema_model, proj_size=self.args.proj_size)
        self.ema_model.load_state_dict(self.model.state_dict())
    

    def init(self, T, p_cutoff, proj_size, K, smoothing_alpha, da_len=0):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.proj_size = proj_size 
        self.K = K
        self.smoothing_alpha = smoothing_alpha
        self.da_len = da_len

        # memeory bank
        self.mem_bank = torch.randn(proj_size, K).cuda(self.gpu)
        self.mem_bank = F.normalize(self.mem_bank, dim=0)
        self.labels_bank = torch.zeros(K, dtype=torch.long).cuda(self.gpu)

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
    def update_bank(self, k, labels, index):
        if self.distributed:
            k = self.concat_all_gather(k)
            labels = self.concat_all_gather(labels)
            index = self.concat_all_gather(index)
        if self.use_ema_teacher:
            self.mem_bank[:, index] = k.t().detach()
        else:
            self.mem_bank[:, index] = F.normalize(self.ema_bank * self.mem_bank[:, index] + (1 - self.ema_bank) * k.t().detach())
        self.labels_bank[index] = labels.detach()
    

    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w.shape[0]
        idx_lb = idx_lb.cuda(self.gpu)

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            bank = self.mem_bank.clone().detach()

            if self.use_cat:
                # inputs = torch.cat((x_lb, x_ulb_s))
                # logits, feats = self.model(inputs)
                # logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
                # logits_x_ulb_s, feats_x_ulb_s = logits[num_lb:], feats[num_lb:]
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                logits, feats = self.model(inputs)
                logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
                ema_logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                ema_feats_x_ulb_w, feats_x_ulb_s = feats[num_lb:].chunk(2)
            else:
                logits_x_lb, ema_feats_x_lb = self.model(x_lb)
                ema_logits_x_ulb_w, ema_feats_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_s, feats_x_ulb_s = self.model(x_ulb_s)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            self.ema.apply_shadow()
            with torch.no_grad():
                # ema teacher model
                if self.use_ema_teacher:
                    _, ema_feats_x_lb = self.model(x_lb)
                ema_probs_x_ulb_w = F.softmax(ema_logits_x_ulb_w, dim=-1)
                if self.da_len:
                    ema_probs_x_ulb_w = self.distribution_alignment(ema_probs_x_ulb_w)
            self.ema.restore()

            with torch.no_grad():
                teacher_logits = ema_feats_x_ulb_w @ bank
                teacher_prob_orig = F.softmax(teacher_logits / self.T, dim=1)
                factor = ema_probs_x_ulb_w.gather(1, self.labels_bank.expand([num_ulb, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if self.smoothing_alpha < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels_bank.expand([bs,-1]) , teacher_prob_orig)
                    probs_x_ulb_w = ema_probs_x_ulb_w * self.smoothing_alpha + aggregated_prob * (1- self.smoothing_alpha)
                else:
                    probs_x_ulb_w = ema_probs_x_ulb_w

            student_logits = feats_x_ulb_s @ bank
            student_prob = F.softmax(student_logits / self.T, dim=1)
            in_loss = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
            if self.epoch == 0:
                in_loss *= 0.0
                probs_x_ulb_w = ema_probs_x_ulb_w

            # compute mask
            max_probs = torch.max(probs_x_ulb_w, dim=-1)[0]
            mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)

            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             probs_x_ulb_w,
                                             'ce',
                                             use_hard_labels=False,
                                             T=1.0,
                                             mask=mask,
                                             softmax=False)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_in * in_loss

            self.update_bank(ema_feats_x_lb, y_lb, idx_lb)

        # parameter updates
        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict
    
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['mem_bank'] = self.mem_bank.cpu()
        save_dict['labels_bank'] = self.labels_bank.cpu()
        if self.da_len:
            save_dict['da_queue'] = self.da_queue.cpu()
            save_dict['da_ptr'] = self.da_ptr.cpu()
        return save_dict
    
    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.mem_bank = checkpoint['mem_bank'].cuda(self.gpu)
        self.labels_bank = checkpoint['labels_bank'].cuda(self.gpu)
        if self.da_len:
            self.da_queue = checkpoint['da_queue'].cuda(self.gpu)
            self.da_ptr = checkpoint['da_ptr'].cuda(self.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--K', int, 128),
            SSL_Argument('--in_loss_ratio', float, 1.0),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
        ]
