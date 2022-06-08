
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
from src.algorithms.algorithmbase import AlgorithmBase
from src.algorithms.utils import ce_loss, consistency_loss, SSL_Argument
from src.datasets.samplers.sampler import DistributedSampler
from src.algorithms.utils import EMA

class Dash(AlgorithmBase):
    """
        Dash algorithm (https://arxiv.org/abs/2109.00650).

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
            - gamma (`float`, *optional*, default=1.27):
                Parameter in computing the dynamic threshold
            - C (`float`, *optional*, default=1.0001):
                Parameter in computing the dynamic threshold
            - rho_min (`float`, *optional*, default=0.05):
                Minimum value of the dynamic threshold
            - num_wu_iter (`int`, *optional*, default=2048):
                Number of warmup iterations
            - num_wu_eval_iter (`int`, *optional*, default=100):
                Number of steps between two evaluations for the warmup phase
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # dash specificed arguments
        self.init(T=args.T, gamma=args.gamma, C=args.C, rho_min=args.rho_min, 
                  num_wu_iter=args.num_wu_iter, num_wu_eval_iter=args.num_wu_eval_iter)
    
    def init(self, T, gamma=1.27, C=1.0001, rho_min=0.05, num_wu_iter=2048, num_wu_eval_iter=100):
        self.T = T 
        self.rho_init = None  # compute from warup training
        self.gamma = gamma 
        self.C = C
        self.rho_min = rho_min
        self.num_wu_iter = num_wu_iter
        self.num_wu_eval_iter = num_wu_eval_iter

        self.rho_init = None
        self.rho_update_cnt = 0
        self.use_hard_label = False
        self.rho = None
        self.warmup_stage = True

    def warmup(self):
        # prevent the training iterations exceed args.num_train_iter
        if self.it > self.num_wu_iter:
            return

        # determine if still in warmup stage
        if not self.warmup_stage:
            self.print_fn("warmup stage finished")
            return

        ngpus_per_node = torch.cuda.device_count()

        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        start_batch.record()
    
        warmup_epoch = max(1, self.num_wu_iter // self.num_iter_per_epoch)

        for epoch in range(warmup_epoch):

             # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_wu_iter:
                break

            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_wu_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                x_lb = data_lb['x_lb']
                y_lb = data_lb['y_lb']
                
                if isinstance(x_lb, dict):
                    x_lb = {k: v.cuda(self.gpu) for k, v in x_lb.items()}
                else:
                    x_lb = x_lb.cuda(self.gpu)
                y_lb = y_lb.cuda(self.gpu)

                # inference and calculate sup/unsup losses
                with self.amp_cm():
                    logits_x_lb = self.model(x_lb)
                    sup_loss = ce_loss(logits_x_lb, y_lb, use_hard_labels=True, reduction='mean')

                # parameter updates
                self.parameter_update(sup_loss)

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                tb_dict = {}
                tb_dict['train/sup_loss'] = sup_loss.item()
                tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                if self.it % self.num_wu_eval_iter == 0:
                    save_path = os.path.join(self.save_dir, self.save_name)
                    if not self.distributed or (self.distributed and self.rank % ngpus_per_node == 0):
                        self.save_model('latest_model.pth', save_path)
                    self.print_fn(f"warmup {self.it} iteration, {tb_dict}")

                del tb_dict
                start_batch.record()
                self.it += 1

        # compute rho_init
        eval_dict = self.evaluate()
        self.rho_init = eval_dict['eval/loss']
        self.rho_update_cnt = 0
        self.use_hard_label = False
        self.rho = self.rho_init
        # reset self it
        self.warmup_stage = False
        self.it = 0
        return


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # adjust rho every 10 epochs
        if self.it % (10 * self.num_iter_per_epoch) == 0:
            self.rho = self.C * (self.gamma ** -self.rho_update_cnt) * self.rho_init
            self.rho = max(self.rho, self.rho_min)
            self.rho_update_cnt += 1
        
        # use hard labels if rho reduced 0.05
        if self.rho == self.rho_min:
            self.use_hard_label = True
        else:
            self.use_hard_label = False

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

            # compute mask
            with torch.no_grad():
                if self.use_hard_label:
                    pseudo_label = torch.argmax(logits_x_ulb_w, dim=-1).detach()
                else:
                    pseudo_label = torch.softmax(logits_x_ulb_w / self.T, dim=-1).detach()
                loss_w = ce_loss(logits_x_ulb_w, pseudo_label, use_hard_labels=self.use_hard_label, reduction='none').detach()
                mask = loss_w.le(self.rho).to(logits_x_ulb_s.dtype).detach()

            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             use_hard_labels=self.use_hard_label,
                                             T=self.T,
                                             mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        # parameter updates
        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict
    
    def get_save_dict(self):
        save_dict =  super().get_save_dict()
        save_dict['rho_init'] = self.rho_init
        save_dict['rho_update_cnt'] = self.rho_update_cnt
        save_dict['rho'] = self.rho
        save_dict['warmup_stage'] = self.warmup_stage
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.rho = checkpoint['rho']
        self.rho_init = checkpoint['rho_init']
        self.warmup_stage = checkpoint['warmup_stage']
        self.rho_update_cnt = checkpoint['rho_update_cnt']
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--gamma', float, 1.27),
            SSL_Argument('--C', float, 1.0001),
            SSL_Argument('--rho_min', float, 0.05),
            SSL_Argument('--num_wu_iter', int, 2048),
            SSL_Argument('--num_wu_eval_iter', int, 100),
        ]
