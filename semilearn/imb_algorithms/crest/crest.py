# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import torch
import numpy as np

from .utils import ProgressiveDistAlignEMAHook, CReSTCheckpointHook, CReSTLoggingHook

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import get_dataset, get_data_loader, send_model_cuda, IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('crest')
class CReST(ImbAlgorithmBase):
    """
        CReST algorithm (https://arxiv.org/abs/2102.09559).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - crest_num_gens (int):
                number of generations for crest
            - crest_dist_align_t (float):
                t parameter in dist align
            - crest_pro_dist_align (bool):
                flag of using progressive dist align
            - crest_alpha (float):
                alpha parameter for crest
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(num_gens=args.crest_num_gens, dist_align_t=args.crest_dist_align_t, pro_dist_align=args.crest_pro_dist_align, sampling_alpha=args.crest_alpha)
        super(CReST, self).__init__(args, net_builder, tb_log, logger, **kwargs)

    def imb_init(self, num_gens=6, dist_align_t=0.5, pro_dist_align=True, sampling_alpha=3):
        self.num_gens = num_gens
        self.dist_align_t = dist_align_t
        self.pro_dist_align = pro_dist_align
        self.sampling_alpha = sampling_alpha
        self.start_gen = 0
        self.pseudo_label_list = None
        self.best_gen = 0 
        self.best_gen_eval_acc = 0.0
    
    def set_hooks(self):
        super().set_hooks()
        
        # reset checkpoint hook
        self.register_hook(CReSTCheckpointHook(), "CheckpointHook", "HIGH")
        self.register_hook(CReSTLoggingHook(), "LoggingHook", "LOW")

        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        lb_class_dist = lb_class_dist / lb_class_dist.sum()
        self.lb_class_dist = lb_class_dist

        # get ground truth distribution
        if self.pro_dist_align:
            self.register_hook(
                ProgressiveDistAlignEMAHook(num_classes=self.num_classes, p_target_type='gt', p_target=lb_class_dist), 
                "DistAlignHook")

    def get_split(self, lb_data, lb_targets, eval_ulb_data, eval_ulb_targets, pseudo_label_list=None):
        if pseudo_label_list is not None and len(pseudo_label_list):
            data_picked = []
            targets_picked = []

            lb_class_dist = self.lb_class_dist
            sorted_class = np.argsort(lb_class_dist)[::-1]
            class_imb_ratio = lb_class_dist[sorted_class][0] / lb_class_dist[sorted_class[-1]]  # self.lb_imb_ratio
            class_imb_ratio = 1. / class_imb_ratio
            mu = np.math.pow(class_imb_ratio, 1 / (self.num_classes - 1))

            for c in sorted_class:
                num_picked = int(
                    len(pseudo_label_list[c]) *
                    np.math.pow(np.math.pow(mu, (self.num_classes - 1) - c), 1 / self.sampling_alpha))  # this is correct!!!
                idx_picked = pseudo_label_list[c][:num_picked]

                try:
                    if len(idx_picked) > 0:
                        data_picked.append(eval_ulb_data[idx_picked])
                        targets_picked.append(np.ones_like(eval_ulb_targets[idx_picked]) * c)
                        print('class {} is added {} pseudo labels'.format(c, num_picked))
                except:
                    continue
            data_picked.append(lb_data)
            targets_picked.append(lb_targets)
            lb_data = np.concatenate(data_picked, axis=0)
            lb_targets = np.concatenate(targets_picked, axis=0)
        else:
            self.print_fn('Labeled data not update')
        return lb_data, lb_targets

    def set_dataset(self, pseudo_label_list=None):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        
        # set include_lb_to_ulb to False
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, include_lb_to_ulb=False)
        # eval_ulb
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
        dataset_dict['eval_ulb'].is_ulb = False

        # add pseudo labels into lb
        lb_data, lb_targets = dataset_dict['train_lb'].data, dataset_dict['train_lb'].targets
        eval_ulb_data, eval_ulb_targets = dataset_dict['eval_ulb'].data, dataset_dict['eval_ulb'].targets
        lb_data, lb_targets = self.get_split(lb_data, lb_targets, eval_ulb_data, eval_ulb_targets, pseudo_label_list)
        dataset_dict['train_lb'].data = lb_data
        dataset_dict['train_lb'].targets = lb_targets

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}, unlabeled eval data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len, len(dataset_dict['eval_ulb'])))
        
        
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict
    
    def set_data_loader(self):
        loader_dict = super().set_data_loader()
        
        # add unlabeled evaluation data loader
        loader_dict['eval_ulb'] = get_data_loader(self.args,
                                                  self.dataset_dict['eval_ulb'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)

        return loader_dict

    def re_init(self):
        self.it = 0
        self.best_gen_eval_acc = 0.0
        self.ema = None
        
        # build dataset with pseudo label list
        self.dataset_dict = self.set_dataset(self.pseudo_label_list)

        # build model and ema_model
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()
        self.model = send_model_cuda(self.args, self.model)
        self.ema_model = send_model_cuda(self.args, self.ema_model)

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build data loader
        self.loader_dict = self.set_data_loader()

    def train(self):

        # EMA Init
        self.model.train()

        for gen in range(self.start_gen, self.num_gens):
            self.gen = gen

            # before train generation
            if self.pro_dist_align:
                cur = self.gen / ( self.num_gens - 1)
                self.cur_dist_align_t = (1.0 - cur) * 1.0 + cur * self.dist_align_t
            else:
                self.cur_dist_align_t = self.dist_align_t

            # re-init every generation
            if self.gen > 0:
                self.re_init()
            
            self.call_hook("before_run")


            for epoch in range(self.start_epoch, self.epochs):
                self.epoch = epoch
                
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_epoch")
                for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                             self.loader_dict['train_ulb']):
                    # prevent the training iterations exceed args.num_train_iter
                    if self.it >= self.num_train_iter:
                        break

                    self.call_hook("before_train_step")
                    # NOTE: progressive dist align will be called inside each train_step in core algorithms
                    self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                    self.call_hook("after_train_step")
                    self.it += 1

                self.call_hook("after_train_epoch")
            
            # after train generation
            eval_dict = {'eval/best_acc': self.best_gen_eval_acc, 'eval/best_it': self.best_it}
            for key, item in eval_dict.items():
                self.print_fn(f"CReST Generation {gen}, Model result - {key} : {item}")

            self.print_fn(f"Generation {self.gen} finished, updating pseudo label list")
            ulb_logits = self.evaluate('eval_ulb', return_logits=True)['eval_ulb/logits']
            if isinstance(ulb_logits, np.ndarray):
                ulb_logits = torch.from_numpy(ulb_logits)
            ulb_score, ulb_pred = torch.max(torch.softmax(ulb_logits, dim=1), dim=1)
            self.pseudo_label_list = []
            for c in range(self.num_classes):
                idx_gather = torch.where(ulb_pred == c)[0]
                if len(idx_gather) == 0:
                    self.pseudo_label_list.append([])
                    continue
                score_gather = ulb_score[idx_gather]
                score_sorted_idx = torch.argsort(score_gather, descending=True)
                idx_gather = idx_gather[score_sorted_idx]
                self.pseudo_label_list.append(idx_gather.numpy())

            self.it = 0 
        
        self.call_hook("after_run")

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['gen'] = self.gen
        if self.pro_dist_align:
            save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
            save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.gen = checkpoint['gen']
        self.start_gen = checkpoint['gen']
        if self.pro_dist_align:
            self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
            self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--crest_num_gens', int, 6),
            SSL_Argument('--crest_dist_align_t', float, 0.5),
            SSL_Argument('--crest_pro_dist_align', str2bool, True),
            SSL_Argument('--crest_alpha', float, 3),
        ]

IMB_ALGORITHMS['crest+'] = CReST