# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from progress.bar import Bar

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from semilearn.core.utils import get_optimizer, get_cosine_schedule_with_warmup, get_logger, EMA



class Trainer:
    def __init__(self, config, algorithm, verbose=0):
        self.config = config
        self.verbose = verbose
        self.algorithm = algorithm

        # TODO: support distributed training?
        torch.cuda.set_device(config.gpu)
        self.algorithm.model = self.algorithm.model.cuda(config.gpu)

        # setup logger
        self.save_path = os.path.join(config.save_dir, config.save_name)
        self.logger = get_logger(config.save_name, save_path=self.save_path, level="INFO")


    def fit(self, train_lb_loader, train_ulb_loader, eval_loader):
        self.algorithm.model.train()

        # EMA Init
        self.algorithm.ema = EMA(self.algorithm.model, self.algorithm.ema_m)
        self.algorithm.ema.register()

        # train
        self.algorithm.it = 0
        self.algorithm.best_eval_acc = 0.0
        self.algorithm.best_epoch = 0

        for epoch in range(self.config.epoch):
            self.algorithm.epoch = epoch
            print("Epoch: {}".format(epoch))
            if self.algorithm.it > self.config.num_train_iter:
                break

            bar = Bar('Processing', max=len(train_lb_loader))

            self.algorithm.model.train()

            for data_lb, data_ulb in zip(train_lb_loader, train_ulb_loader):

                if self.algorithm.it > self.config.num_train_iter:
                    break
                
                result = self.algorithm.train_step(**self.algorithm.process_batch(**data_lb, **data_ulb))

                bar.suffix = ("Iter: {batch:4}/{iter:4}.".format(batch=self.algorithm.it, iter=len(train_lb_loader)))
                bar.next()
                self.algorithm.it += 1
            bar.finish()

            # validate
            result = self.evaluate(eval_loader)

            # save model
            self.algorithm.save_model('latest_model.pth', self.save_path)

            # best
            if result['acc'] > self.algorithm.best_eval_acc:
                self.algorithm.best_eval_acc = result['acc']
                self.algorithm.best_epoch = self.algorithm.epoch
                self.algorithm.save_model('model_best.pth', self.save_path)
        
        self.logger.info("Best acc {:.4f} at epoch {:d}".format(self.algorithm.best_eval_acc, self.algorithm.best_epoch))
        self.logger.info("Training finished.")


    def evaluate(self, data_loader, use_ema_model=False):
        y_pred, y_logits, y_true = self.predict(data_loader, use_ema_model, return_gt=True)
        top1 = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.logger.info("confusion matrix")
        self.logger.info(cf_mat)
        result_dict = {'acc': top1, 'precision': precision, 'recall': recall, 'f1': f1}
        self.logger.info("evaluation metric")
        for key, item in result_dict.items():
            self.logger.info("{:s}: {:.4f}".format(key, item))
        return result_dict


    def predict(self, data_loader, use_ema_model=False, return_gt=False):
        self.algorithm.model.eval()
        if use_ema_model:
            self.algorithm.ema.apply_shadow()

        y_true = []
        y_pred = []
        y_logits = []
        with torch.no_grad():
            for data in data_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.config.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.config.gpu)
                y = y.cuda(self.config.gpu)

                logits = self.algorithm.model(x)['logits']
                    
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        if use_ema_model:
            self.algorithm.ema.restore()
        self.algorithm.model.train()
        
        if return_gt:
            return y_pred, y_logits, y_true
        else:
            return y_pred, y_logits