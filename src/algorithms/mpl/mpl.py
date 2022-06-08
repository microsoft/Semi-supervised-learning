# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from src.utils import get_optimizer, get_cosine_schedule_with_warmup
from copy import deepcopy
from src.algorithms.algorithmbase import AlgorithmBase
from src.algorithms.utils import ce_loss, consistency_loss, Get_Scalar, smooth_targets, SSL_Argument, str2bool


class MPL(AlgorithmBase):
    """
        Meta Pseudo Label algorithm (https://arxiv.org/abs/2003.10580).
        Reference implementation (https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).

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
            - label_smoothing ('float'):
                Label smoothing value for cross-entropy loss
            - tsa_schedule ('str'):
                TSA schedule to use
            - num_uda_warmup_iter ('int'):
                Number of iterations for uda to warmup
            - num_stu_wait_iter ('int):
                Number of iterations for student network waiting until starting training
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, p_cutoff=args.p_cutoff, 
                  label_smoothing=args.label_smoothing, tsa_schedule=args.tsa_schedule,
                  num_uda_warmup_iter=args.num_uda_warmup_iter, num_stu_wait_iter=args.num_stu_wait_iter)
        self.build_teacher_model(net_builder)

    def init(self, T, p_cutoff, tsa_schedule, label_smoothing, num_uda_warmup_iter, num_stu_wait_iter):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.tsa_schedule = tsa_schedule
        # self.use_hard_label = hard_label
        self.label_smoothing = label_smoothing
        self.num_uda_warmup_iter = num_uda_warmup_iter
        self.num_stu_wait_iter = num_stu_wait_iter
        self.moving_dot_product = torch.zeros(1).cuda(self.gpu)

    def build_teacher_model(self, net_builder):
        # create teacher model 
        self.teacher_model = net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
        self.teacher_optimizer = get_optimizer(self.teacher_model, self.args.optim, self.args.teacher_lr, self.args.momentum, self.args.weight_decay)
        self.teacher_scheduler = get_cosine_schedule_with_warmup(self.teacher_optimizer, self.args.num_train_iter, num_warmup_steps=self.args.num_train_iter * 0)
        if not torch.cuda.is_available():
            raise Exception('ONLY GPU TRAINING IS SUPPORTED')
        elif self.args.distributed:
            if self.args.gpu is not None:
                self.teacher_model = self.teacher_model.cuda(self.args.gpu)
                self.teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher_model)
                self.teacher_model = torch.nn.parallel.DistributedDataParallel(self.teacher_model,
                                                                               device_ids=[self.args.gpu],
                                                                               broadcast_buffers=False)
            else:
                # if arg.gpu is None, DDP will divide and allocate batch_size
                # to all available GPUs if device_ids are not set.
                self.teacher_model = self.teacher_model.cuda(self.gpu)
                self.teacher_model = torch.nn.parallel.DistributedDataParallel(self.teacher_model)
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu)
            self.teacher_model = self.teacher_model.cuda(self.args.gpu)

        else:
            self.teacher_model = torch.nn.DataParallel(self.teacher_model).cuda(self.gpu)
        self.teacher_loss_scaler = GradScaler()

    def teacher_parameter_update(self, loss):
         # parameter updates
        if self.use_amp:
            self.teacher_loss_scaler.scale(loss).backward()
            if (self.clip_grad > 0):
                self.teacher_loss_scaler.unscale_(self.teacher_optimizer)
                torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), self.clip_grad)
            self.teacher_loss_scaler.step(self.teacher_optimizer)
            self.teacher_loss_scaler.update()
        else:
            loss.backward()
            if (self.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), self.clip_grad)
            self.teacher_optimizer.step()

        self.teacher_scheduler.step()
        self.teacher_model.zero_grad()


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                logits = self.teacher_model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            else:
                logits_x_lb = self.teacher_model(x_lb)
                logits_x_ulb_w = self.teacher_model(x_ulb_w)
                logits_x_ulb_s = self.teacher_model(x_ulb_s)

            # hyper-params for update
            tsa = self.TSA(self.tsa_schedule, self.it, self.num_train_iter, self.num_classes)  # Training Signal Annealing
            sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()
            if self.label_smoothing:
                targets_x_lb = smooth_targets(logits_x_lb, y_lb, self.label_smoothing)
                use_hard_labels = False
            else:
                targets_x_lb = y_lb
                use_hard_labels = True
            sup_loss = (ce_loss(logits_x_lb, targets_x_lb, use_hard_labels, reduction='none') * sup_mask).mean()

            # compute mask
            max_probs = torch.max(torch.softmax(logits_x_ulb_w.detach(), dim=-1), dim=-1)[0]
            mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)

            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             # TODO: check this 
                                             use_hard_labels=False,
                                             T=self.T,
                                             mask=mask,
                                             label_smoothing=self.label_smoothing)

            # 1st call to student
            if self.use_cat:
                inputs = torch.cat([x_lb, x_ulb_s], dim=0)
                logits = self.model(inputs)
                s_logits_x_lb_old = logits[:num_lb]
                s_logits_x_ulb_s = logits[num_lb:]
            else:
                s_logits_x_lb_old = self.model(x_lb)
                s_logits_x_ulb_s = self.model(x_ulb_s)

            s_max_probs = torch.max(torch.softmax(logits_x_ulb_s.detach(), dim=-1), dim=-1)[0]
            s_mask = s_max_probs.ge(self.p_cutoff).to(s_max_probs.dtype)

            # update student on unlabeled data
            s_unsup_loss, _ = consistency_loss(s_logits_x_ulb_s,
                                               logits_x_ulb_s,
                                               'ce', 
                                               # TODO: check this
                                               use_hard_labels=False,
                                               T=self.T, 
                                               mask=s_mask,
                                               label_smoothing=self.label_smoothing)
            

        # update student's parameters
        self.parameter_update(s_unsup_loss)

        # 2nd call to student
        with self.amp_cm():
            s_logits_x_lb_new = self.model(x_lb)

            # compute teacher's feedback coefficient
            s_sup_loss_old = F.cross_entropy(s_logits_x_lb_old.detach(), y_lb)
            s_sup_loss_new = F.cross_entropy(s_logits_x_lb_new.detach(), y_lb)
            dot_product = s_sup_loss_old - s_sup_loss_new
            self.moving_dot_product = self.moving_dot_product * 0.99 + dot_product * 0.01
            dot_product = dot_product - self.moving_dot_product
            dot_product = dot_product.detach()

            # compute mpl loss
            _, hard_pseudo_label = torch.max(logits_x_ulb_s.detach(), dim=-1)
            mpl_loss = dot_product * ce_loss(logits_x_ulb_s, hard_pseudo_label).mean()
            
            # compute total loss for update teacher
            weight_u = self.lambda_u * min(1., (self.it+1) / self.num_uda_warmup_iter)
            total_loss = sup_loss + weight_u * unsup_loss + mpl_loss

        # update teacher's parameters
        self.teacher_parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/s_unsup_loss'] = s_unsup_loss.item()
        tb_dict['train/mpl_loss'] = mpl_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict

    def TSA(self, schedule, cur_iter, total_iter, num_classes):
        training_progress = cur_iter / total_iter

        if schedule == 'linear':
            threshold = training_progress
        elif schedule == 'exp':
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == 'log':
            scale = 5
            threshold = 1 - math.exp((-training_progress) * scale)
        elif schedule == 'none':
            return 1
        tsa = threshold * (1 - 1 / num_classes) + 1 / num_classes
        return tsa
    
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['moving_dot_product'] = self.moving_dot_product.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.moving_dot_product = checkpoint['moving_dot_product'].cuda(self.gpu)
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--tsa_schedule', str, 'none', 'TSA mode: none, linear, log, exp'),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--teacher_lr', float, 0.03),
            SSL_Argument('--label_smoothing', float, 0.1),
            SSL_Argument('--num_uda_warmup_iter', int, 5000),
            SSL_Argument('--num_stu_wait_iter', int, 3000)
        ]
