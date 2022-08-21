# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.algorithmbase import AlgorithmBase
from semilearn.algorithms.utils import ce_loss, EMA
from semilearn.datasets import DistributedSampler


class FullySupervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def train_step(self, x_lb, y_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

        # parameter updates
        self.parameter_update(sup_loss)

        # tensorboard_dict update
        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        return tb_dict

    
    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if self.resume == True:
            self.ema.load(self.ema_model)
            eval_dict = self.evaluate()
            self.print_fn(eval_dict)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        start_batch.record()
            
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                self.tb_dict = self.train_step(**self.process_batch(**data_lb))

                end_run.record()
                torch.cuda.synchronize()

                self.tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                self.tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                self.tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                self.after_train_step()
                start_batch.record()

        eval_dict = self.evaluate()
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        return eval_dict

    @staticmethod
    def get_argument():
        return {}
