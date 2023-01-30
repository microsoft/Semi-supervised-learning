# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from cProfile import label
from collections import defaultdict
import torch

from semilearn.algorithms.hooks import PseudoLabelingHook


class DASOFeatureQueue:
    def __init__(self, num_classes, feat_dim, queue_length, classwise_max_size=None, bal_queue=True):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.max_size = queue_length

        self.bank = defaultdict(lambda: torch.empty(0, self.feat_dim))
        self.prototypes = torch.zeros(self.num_classes, self.feat_dim)
        self.classwise_max_size = classwise_max_size
        self.bal_queue = bal_queue

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor): 
        if not self.prototypes.is_cuda:
            self.prototypes = self.prototypes.to(features.device)

        for idx in range(self.num_classes):
            
            # per class max size
            max_size = (self.classwise_max_size[idx] * 5) if self.classwise_max_size is not None else self.max_size
            if self.bal_queue:
                max_size = self.max_size

            # select features by label
            cls_inds = torch.where(labels == idx)[0]
            if len(cls_inds):
                with torch.no_grad():
                    # push to the memory bank
                    feats_selected = features[cls_inds]
                    self.bank[idx] = torch.cat([self.bank[idx], feats_selected.cpu()], 0)

                    # fixed size
                    current_size = len(self.bank[idx])
                    if current_size > max_size:
                        self.bank[idx] = self.bank[idx][current_size - max_size:]

                    # update prototypes
                    self.prototypes[idx, :] = self.bank[idx].mean(0).to(features.device)
    

class DASOPseudoLabelingHook(PseudoLabelingHook):
    def __init__(self, num_classes, T_dist, with_dist_aware, interp_alpha):
        super().__init__()
        self.num_classes = num_classes
        self.T_dist = T_dist
        self.with_dist_aware = with_dist_aware
        self.interp_alpha = interp_alpha

        self.pseudo_label_list = []
        self.pseudo_label_dist = [0 for i in range(self.num_classes)] 

    def push_pl_list(self, pl_list):
        self.pseudo_label_list.append(pl_list)

    def update_pl_dist(self):
        if len(self.pseudo_label_list) == 0: return

        pl_total_list = torch.cat(self.pseudo_label_list, 0)
        for class_ind in range(self.num_classes):
            pl_row_inds = torch.where(pl_total_list == class_ind)[0]
            self.pseudo_label_dist[class_ind] = len(pl_row_inds)
        self.pseudo_label_list = []

    def get_pl_dist(self, normalize=True):
        if isinstance(self.pseudo_label_dist, list):
            pl_dist = torch.Tensor(self.pseudo_label_dist).float()
        else:
            pl_dist = self.pseudo_label_dist.float()
        if normalize:
            pl_dist = pl_dist / pl_dist.sum()
        return pl_dist


    @torch.no_grad()
    def gen_ulb_targets(self, 
                        algorithm, 
                        logits, 
                        use_hard_label=True, 
                        T=1.0,
                        softmax=True, # whether to compute softmax for logits, input must be logits
                        label_smoothing=0.0):

        if algorithm.it  < algorithm.num_pretrain_iter:
            pseudo_label = torch.argmax(logits.detach(), dim=-1)
            # update pseudo label list
            self.push_pl_list(pseudo_label.detach().cpu())
            return super().gen_ulb_targets(algorithm=algorithm, logits=logits, use_hard_label=use_hard_label, T=T, softmax=softmax, label_smoothing=label_smoothing)

        logits = logits.detach()
        if softmax:
            # probs = torch.softmax(logits, dim=1)
            probs = self.compute_prob(logits)
        else:
            probs = logits
        pseudo_label = torch.argmax(probs.detach(), dim=-1)

        # compute the mix
        current_pl_dist = self.get_pl_dist().to(probs.device)  # (1, C)
        current_pl_dist = current_pl_dist**(1. / algorithm.T_dist)
        current_pl_dist = current_pl_dist / current_pl_dist.sum()
        current_pl_dist = current_pl_dist / current_pl_dist.max()  # MIXUP
        
        pred_to_dist = current_pl_dist[pseudo_label].view(-1, 1)  # (B, )

        if not algorithm.with_dist_aware:
            pred_to_dist = algorithm.interp_alpha  # override to fixed constant

        # pl mixup
        probs_mixup = (1. - pred_to_dist) * probs + pred_to_dist * algorithm.probs_sim

        pseudo_label = torch.argmax(probs_mixup.detach(), dim=-1) 

        # update pseudo label list
        self.push_pl_list(pseudo_label.detach().cpu())

        if use_hard_label:
            return pseudo_label
        else:
            return probs_mixup
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_pl_dist_iter):
            self.update_pl_dist()