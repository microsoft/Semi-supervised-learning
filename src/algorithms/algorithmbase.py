# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import contextlib
import numpy as np
from inspect import signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from src.datasets.samplers.sampler import DistributedSampler
from src.algorithms.utils import Bn_Controller, EMA


class AlgorithmBase:
    """
        Base class for algorithms
        init algorithm specific parameters and common parameters
        
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
    def __init__(
        self,
        args,
        net_builder,
        tb_log=None,
        logger=None,
        **kwargs):
        
        # common arguments
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio 
        self.use_cat = args.use_cat
        self.use_amp = args.use_amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        # commaon utils arguments
        self.tb_log = tb_log
        self.logger = logger 
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed

        # common model related parameters
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.optimizer = None
        self.scheduler = None
        self.loader_dict = {}
        
        # cv, nlp, speech builder different arguments
        self.model = net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
        self.net_builder = net_builder
        self.ema_model = self.set_ema_model()
        self.ema = None

        # other arguments specific to this algorithm
        # self.init(**kwargs)

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def process_batch(self, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        input_args = signature(self.train_step).parameters
        input_args = list(input_args.keys())
        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue
            
            if var is None:
                continue
            
            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict


    def parameter_update(self, loss):
        """
        # parameter updates
        """
        if self.use_amp:
            self.loss_scaler.scale(loss).backward()
            if (self.clip_grad > 0):
                self.loss_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
        else:
            loss.backward()
            if (self.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

        self.scheduler.step()
        self.model.zero_grad()
        if self.ema is not None:
            self.ema.update()

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record tb_dict
        # return tb_dict
        raise NotImplementedError


    def before_train_step(self):
        raise NotImplementedError

    def after_train_step(self):
        """
        save model and printing log
        """
        # Save model 
        if (self.it + 1) % self.num_eval_iter == 0:
            save_path = os.path.join(self.save_dir, self.save_name)
            if not self.distributed or (self.distributed and self.rank % self.ngpus_per_node == 0):
                self.save_model('latest_model.pth', save_path)

        if (self.it + 1) % self.num_eval_iter == 0:
            eval_dict = self.evaluate('eval')
            self.tb_dict.update(eval_dict)

            save_path = os.path.join(self.save_dir, self.save_name)

            if self.tb_dict['eval/top-1-acc'] > self.best_eval_acc:
                self.best_eval_acc = self.tb_dict['eval/top-1-acc']
                self.best_it = self.it

            if not self.distributed or (self.distributed and self.rank % self.ngpus_per_node == 0):
                self.print_fn(f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {self.tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_it} iters")


            if not self.distributed or  (self.distributed and self.rank % self.ngpus_per_node == 0):

                if self.it == self.best_it:
                    self.save_model('model_best.pth', save_path)

                if not self.tb_log is None:
                    self.tb_log.update(self.tb_dict, self.it)
        
        self.it += 1
        del self.tb_dict
        if self.it > 0.9 * self.num_train_iter:
            self.num_eval_iter = 1024

    def train(self):
        """
        train function
        """

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if self.resume == True:
            self.ema.load(self.ema_model)
            # eval_dict = self.evaluate()
            # self.print_fn(eval_dict)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        start_batch.record()


        for epoch in range(self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)
            if isinstance(self.loader_dict['train_ulb'].sampler, DistributedSampler):
                self.loader_dict['train_ulb'].sampler.set_epoch(epoch)

            # for (idx_lb, x_lb, y_lb), (idx_ulb, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
            #                                                              self.loader_dict['train_ulb']):
            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                # self.tb_dict = self.train_step(**self.process_batch(idx_lb=idx_lb, x_lb=x_lb, y_lb=y_lb, idx_ulb=idx_ulb, x_ulb_w=x_ulb_w, x_ulb_s=x_ulb_s))
                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                self.tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                self.tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                self.tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                # post processing for saving model and print logs
                self.after_train_step()
                start_batch.record()

        # eval_dict = self.evaluate('eval')
        eval_dict = {'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it}
        if 'test' in self.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(self.args.save_dir, self.args.save_name, 'model_best.pth')
            self.load_model(best_model_path)
            test_dict = self.evaluate('test')
            eval_dict['test/best_acc'] = test_dict['test/top-1-acc']
        return eval_dict

    def evaluate(self, eval_dest='eval'):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()
        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch
                if self.algorithm in ['crmatch', 'comatch', 'simmatch']:
                    logits, *_ = self.model(x)
                else:
                    logits = self.model(x)
                loss = F.cross_entropy(logits, y, reduction='mean')
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        # top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        # top5 = 0
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest+'/loss': total_loss / total_num, eval_dest+'/top-1-acc': top1, 
                     eval_dest+'/precision': precision, eval_dest+'/recall': recall, eval_dest+'/F1': F1}
        return eval_dict

    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        save_dict = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'it': self.it + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
        }
        return save_dict
    

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        self.ema_model_state_dict = self.model.state_dict()
        self.ema.restore()
        self.model.train()
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")


    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.ema_model.load_state_dict(self.check_prefix_state_dict(checkpoint['ema_model']))
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.it = checkpoint['it']
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']
        self.print_fn('model loaded')
        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        raise NotImplementedError
