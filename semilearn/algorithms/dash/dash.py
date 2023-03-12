
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
from .utils import DashThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import EMA, ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument
from semilearn.datasets import DistributedSampler


@ALGORITHMS.register('dash')
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
        # dash specified arguments
        self.init(T=args.T, num_wu_iter=args.num_wu_iter, num_wu_eval_iter=args.num_wu_eval_iter)
    
    def init(self, T, num_wu_iter=2048, num_wu_eval_iter=100):
        self.T = T 
        self.num_wu_iter = num_wu_iter
        self.num_wu_eval_iter = num_wu_eval_iter
        self.use_hard_label = False
        self.warmup_stage = True

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(DashThresholdingHook(rho_min=self.args.rho_min, gamma=self.args.gamma, C=self.args.C), "MaskingHook")
        super().set_hooks()

    def warmup(self):
        # TODO: think about this, how to make this compatible with hooks?
        
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
                    logits_x_lb = self.model(x_lb)['logits']
                    sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

                self.out_dict = {'loss': sup_loss}
                # parameter updates     
                # self.parameter_update(sup_loss)
                self.call_hook("after_train_step", "ParamUpdateHook")

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                log_dict = {}
                log_dict['train/sup_loss'] = sup_loss.item()
                log_dict['lr'] = self.optimizer.param_groups[0]['lr']
                log_dict['train/prefetch_time'] = start_batch.elapsed_time(end_batch) / 1000.
                log_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                if self.it % self.num_wu_eval_iter == 0:
                    # save_path = os.path.join(self.save_dir, self.save_name)
                    # if not self.distributed or (self.distributed and self.rank % ngpus_per_node == 0):
                    #     self.save_model('latest_model.pth', save_path)
                    self.print_fn(f"warmup {self.it} iteration, {log_dict}")

                del log_dict
                start_batch.record()
                self.it += 1

        # compute rho_init
        eval_dict = self.evaluate()
        self.rho_init = eval_dict['eval/loss']
        # self.rho_update_cnt = 0
        # self.use_hard_label = False
        self.rho = self.rho_init
        # reset self it
        self.warmup_stage = False
        self.it = 0
        return


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
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

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

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
    
    def get_save_dict(self):
        save_dict =  super().get_save_dict()
        save_dict['rho_init'] = self.hooks_dict['MaskingHook'].rho_init
        save_dict['rho_update_cnt'] = self.hooks_dict['MaskingHook'].rho_update_cnt
        save_dict['rho'] = self.hooks_dict['MaskingHook'].rho
        save_dict['warmup_stage'] = self.warmup_stage
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].rho = checkpoint['rho']
        self.hooks_dict['MaskingHook'].rho_init = checkpoint['rho_init']
        self.warmup_stage = checkpoint['warmup_stage']
        self.hooks_dict['MaskingHook'].rho_update_cnt = checkpoint['rho_update_cnt']
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
