# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np

from semilearn.core.hooks import Hook
from semilearn.core.utils import get_data_loader
from semilearn.core.criterions import CELoss


class EffectiveDistribution(Hook):
    def after_train_epoch(self, algorithm):
        # evaluate on unlabeled data
        logits = algorithm.evaluate('eval_ulb', return_logits=True)['eval_ulb/logits']
        probs = torch.from_numpy(logits).softmax(dim=-1)
        max_probs, pseudo_labels = torch.max(probs, dim=-1)
        
        # get pseudo_labels, sorted by small entropy to large entropy
        pseudo_label_list = []
        for c in range(algorithm.num_classes):
            idx_gather = torch.where(pseudo_labels == c)[0]
            if len(idx_gather) == 0:
                pseudo_label_list.append([])
                continue
            score_gather = max_probs[idx_gather]
            score_sorted_idx = torch.argsort(score_gather, descending=True)
            idx_gather = idx_gather[score_sorted_idx]
            pseudo_label_list.append(idx_gather.numpy().tolist())
        pseudo_labels = pseudo_labels.numpy()


        # update algorithm lb dataset
        dataset_dict = algorithm.set_dataset()
        train_lb_dataset = dataset_dict['train_lb']
        eval_ulb_dataset = algorithm.dataset_dict['eval_ulb']
        train_lb_data, train_lb_targets = train_lb_dataset.data, train_lb_dataset.targets
        eval_ulb_data = np.array(eval_ulb_dataset.data)

        # compute append numbers
        lb_class_dist = algorithm.lb_class_dist
        algorithm.print_fn("lb class dist.: {}".format(lb_class_dist))
        
        # which to append
        ext_ratio =  -lb_class_dist + lb_class_dist.max()
        algorithm.print_fn("ext. ratio: {}".format(ext_ratio))

        # how much to append
        base_number = (len(train_lb_dataset) / algorithm.num_classes)
        ext_length = ext_ratio * base_number

        # annealing
        alpha = 1 - (algorithm.epoch / algorithm.epochs - 1) ** 2
        ext_length = ext_length * alpha
        print("ext length {}".format(ext_length))

        ext_data = []
        ext_targets = []
        for c in range(algorithm.num_classes):
            idx_gather = pseudo_label_list[c][:int(ext_length[c])]
            if len(idx_gather) == 0:
                continue
            idx_gather = np.array(idx_gather).reshape(-1).astype(np.int64)
            if len(idx_gather) == 1:
                idx_gather = idx_gather[0]
                ext_data.append([eval_ulb_data[idx_gather]])
                ext_targets.append([pseudo_labels[idx_gather]])
            else:
                ext_data.append(eval_ulb_data[idx_gather])
                ext_targets.append(pseudo_labels[idx_gather])
        
        if len(ext_data) > 0:
            ext_data = np.concatenate(ext_data, axis=0)
            ext_targets = np.concatenate(ext_targets, axis=0)

            train_lb_dataset.data = np.concatenate([train_lb_data, ext_data], axis=0)
            train_lb_dataset.targets = np.concatenate([train_lb_targets, ext_targets], axis=0)
            algorithm.dataset_dict['train_lb'] = train_lb_dataset
            y_lb_cnt = [0 for _ in range(algorithm.num_classes)]
            for c in train_lb_dataset.targets:
                y_lb_cnt[c] += 1
            y_lb_cnt = torch.from_numpy(np.array(y_lb_cnt)).to(algorithm.gpu)
            algorithm.print_fn("lb count {}".format(y_lb_cnt))
            algorithm.print_fn("lb number {}".format(y_lb_cnt.sum()))
            algorithm.lb_class_dist = y_lb_cnt / y_lb_cnt.sum()
            if algorithm.args.simis_la:
                algorithm.ce_loss.update_lb_dist(algorithm.lb_class_dist)
            algorithm.loader_dict['train_lb'] =  get_data_loader(algorithm.args,
                                                    train_lb_dataset,
                                                    algorithm.args.batch_size,
                                                    data_sampler=algorithm.args.train_sampler,
                                                    num_iters=algorithm.num_train_iter,
                                                    num_epochs=algorithm.epochs,
                                                    num_workers=algorithm.args.num_workers,
                                                    distributed=algorithm.distributed)
        


class LogitsAdjCELoss(CELoss):
    def __init__(self, lb_class_dist):
        super().__init__()
        self.lb_class_dist = lb_class_dist

    def update_lb_dist(self, lb_class_dist):
        self.lb_class_dist = lb_class_dist

    def forward(self, logits, targets, reduction='mean'):
        return super().forward(logits + torch.log(self.lb_class_dist + 1e-12), targets, reduction)
