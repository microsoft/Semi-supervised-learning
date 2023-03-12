# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.hooks import ParamUpdateHook
from semilearn.core.utils import get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument

from .utils import classifier_warmup, get_weighted_sampler, make_imb_data


class CoSSL_Net(nn.Module):
    def __init__(self, backbone, num_classes):
        super(CoSSL_Net, self).__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        if hasattr(backbone, 'backbone'):
            self.classifier = backbone.backbone.classifier
        else:
            self.classifier = backbone.classifier
        self.teacher_classifier = nn.Linear(self.num_features, num_classes, bias=True)

    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        feat = results_dict['feat']
        logits = self.classifier(feat)
        tfe_logits = self.teacher_classifier(feat)
        # logits += 0 * tfe_logits # todo: stupid workaround
        results_dict['logits'] = logits 
        results_dict['logits_tfe'] = tfe_logits
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class CoSSLParamUpdateHook(ParamUpdateHook):
    def after_train_step(self, algorithm):
        loss = algorithm.out_dict['loss']
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()
        
        # cossl teacher classifier update
        if algorithm.epoch >= algorithm.warm_epoch:
            algorithm.tfe_optimizer.step()

        if algorithm.scheduler is not None:
            algorithm.scheduler.step()
        algorithm.model.zero_grad()

        algorithm.end_run.record()
        torch.cuda.synchronize()
        algorithm.log_dict['lr'] = algorithm.optimizer.param_groups[-1]['lr']
        algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.
    

    def before_train_epoch(self, algorithm):
        if algorithm.epoch == algorithm.warm_epoch:
            # initialize teacher classifier
            init_teacher, init_ema_teacher = classifier_warmup(algorithm.args, copy.deepcopy(algorithm.ema_model),
                                                               algorithm.dataset_dict['train_lb'],
                                                               algorithm.dataset_dict['train_ulb'],
                                                               algorithm.lb_cnt_per_class, algorithm.num_classes,
                                                               algorithm.gpu)
            
            algorithm.model.module.teacher_classifier.weight.data.copy_(init_teacher.module.classifier.weight.data)
            algorithm.model.module.teacher_classifier.bias.data.copy_(init_teacher.module.classifier.bias.data)
            algorithm.ema_model.module.teacher_classifier.weight.data.copy_(init_ema_teacher.module.classifier.weight.data)
            algorithm.ema_model.module.teacher_classifier.bias.data.copy_(init_ema_teacher.module.classifier.bias.data)
            algorithm.ema.load(algorithm.ema_model)

            algorithm.mixup_prob = [(max(algorithm.lb_cnt_per_class) - i) / max(algorithm.lb_cnt_per_class) for i in algorithm.lb_cnt_per_class]


@IMB_ALGORITHMS.register('cossl')
class CoSSL(ImbAlgorithmBase):
    """
        CoSSL algorithm (https://arxiv.org/abs/2112.04564).

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
    def __init__(self, args, **kwargs):
        self.imb_init(max_lam=args.cossl_max_lam, tfe_augment=args.cossl_tfe_augment,
                      tfe_u_ratio=args.cossl_tfe_u_ratio, warm_epoch=args.cossl_warm_epoch)
        super().__init__(args, **kwargs)

        self.model = CoSSL_Net(self.model, num_classes=self.num_classes)
        self.ema_model = CoSSL_Net(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()
        self.tfe_optimizer = get_optimizer(self.model.teacher_classifier, self.args.optim, self.args.cossl_tfe_warm_lr, self.args.momentum, self.args.cossl_tfe_warm_wd, 1.0)


    def imb_init(self, max_lam, tfe_augment, tfe_u_ratio, warm_epoch):
        self.max_lam = max_lam
        self.tfe_augment = tfe_augment
        self.tfe_u_ratio = tfe_u_ratio
        self.warm_epoch = warm_epoch
    
    def set_data_loader(self):
        loader_dict = super().set_data_loader()

        # get lb_cnt 
        lb_cnt_per_class = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_cnt_per_class[c] += 1
        lb_cnt_per_class = np.array(lb_cnt_per_class)
        self.lb_cnt_per_class = lb_cnt_per_class

        tfe_labeled_set = copy.deepcopy(self.dataset_dict['train_lb'])
        tfe_unlabeled_set = copy.deepcopy(self.dataset_dict['train_ulb'])

        if self.tfe_augment == 'weak':
            tfe_labeled_set.transform = self.dataset_dict['train_ulb'].transform
        elif self.tfe_augment == 'strong':
            tfe_labeled_set.transform = self.dataset_dict['train_ulb'].strong_transform
        else:
            raise NotImplementedError
        tfe_unlabeled_set.transform = tfe_labeled_set.transform

        # TODO: better to use our own get_data_loader
        tfe_unlabeled_loader = data.DataLoader(tfe_unlabeled_set, batch_size=self.tfe_u_ratio * self.args.batch_size,
                                               shuffle=True, num_workers=0, drop_last=True)

        # TODO: better to use our own weighted sampler
        class_balanced_disb = torch.Tensor(make_imb_data(30000, self.num_classes, 1))
        class_balanced_disb = class_balanced_disb / class_balanced_disb.sum()
        sampler_x = get_weighted_sampler(class_balanced_disb, torch.Tensor(self.lb_cnt_per_class),
                                         tfe_labeled_set.targets)
        
        # TODO: better to use our own get_data_loader
        tfe_labeled_loader = data.DataLoader(tfe_labeled_set, batch_size=self.args.batch_size,
                                             sampler=sampler_x, drop_last=True, num_workers=0)
        
        loader_dict['tfe_train_lb'] = tfe_labeled_loader
        loader_dict['tfe_train_ulb'] = tfe_unlabeled_loader
        return loader_dict


    def set_hooks(self):
        super().set_hooks()

        # reset ParamUpdateHook hook, CoSSL training code is implemented here
        self.register_hook(CoSSLParamUpdateHook(), "ParamUpdateHook", "HIGHEST")

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):
        if self.epoch < self.warm_epoch:
            return super().train_step(*args, **kwargs)

        out_dict, log_dict = super().train_step(*args, **kwargs)

        # CoSSL code starts:
        try:
            labeled_dict = next(self.tfe_labeled_iter)
            tfe_input_x = labeled_dict['x_lb']
            tfe_targets_x = labeled_dict['y_lb']
        except:
            self.tfe_labeled_iter = iter(self.loader_dict['tfe_train_lb'])
            labeled_dict = next(self.tfe_labeled_iter)
            tfe_input_x = labeled_dict['x_lb']
            tfe_targets_x = labeled_dict['y_lb']

        try:
            unlabeled_dict = next(self.tfe_unlabeled_iter)
            if self.args.algorithm in ['remixmatch', 'comatch']:
                tfe_input_u = unlabeled_dict['x_ulb_s_0']
            else:
                tfe_input_u = unlabeled_dict['x_ulb_s']
        except:
            self.tfe_unlabeled_iter = iter(self.loader_dict['tfe_train_ulb'])
            unlabeled_dict = next(self.tfe_unlabeled_iter)
            if self.args.algorithm in ['remixmatch', 'comatch']:
                tfe_input_u = unlabeled_dict['x_ulb_s_0']
            else:
                tfe_input_u = unlabeled_dict['x_ulb_s']

        tfe_input_x = tfe_input_x.cuda(self.gpu)
        tfe_input_u = tfe_input_u.cuda(self.gpu)
        tfe_targets_x = tfe_targets_x.cuda(self.gpu)

        with torch.no_grad():
            tfe_feat_x = self.ema_model(tfe_input_x)['feat']
            tfe_feat_x = tfe_feat_x.squeeze()

            tfe_feat_u = self.ema_model(tfe_input_u)['feat']
            tfe_feat_u = tfe_feat_u.squeeze()

            new_feat_list = []
            new_target_list = []
            for x, label_x, u in zip(tfe_feat_x, tfe_targets_x, tfe_feat_u[:len(tfe_targets_x)]):
                if random.random() < self.mixup_prob[label_x.argmax()]:
                    lam = np.random.uniform(self.max_lam, 1., size=1)
                    lam = torch.FloatTensor(lam).cuda(self.gpu)

                    new_feat = lam * x + (1 - lam) * u
                    new_target = label_x
                else:
                    new_feat = x
                    new_target = label_x
                new_feat_list.append(new_feat)
                new_target_list.append(new_target)
            new_feat_tensor = torch.stack(new_feat_list, dim=0)  # [64, 128]
            new_target_tensor = torch.stack(new_target_list, dim=0)  # [64, 10]

        teacher_logits = self.model.module.teacher_classifier(new_feat_tensor)
        teacher_loss = self.ce_loss(teacher_logits, new_target_tensor, reduction='mean')
 
        out_dict['loss'] += teacher_loss
        log_dict['train/tea_loss'] = teacher_loss.item()
        return out_dict, log_dict


    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        if self.epoch >= self.warm_epoch:
            out_key = 'logits_tfe'
        else:
            out_key = 'logits'
        return super().evaluate(eval_dest=eval_dest, out_key=out_key, return_logits=return_logits)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--cossl_max_lam', float, 0.6),
            SSL_Argument('--cossl_tfe_augment', str, 'strong'),
            SSL_Argument('--cossl_tfe_u_ratio', int, 1),

            SSL_Argument('--cossl_warm_epoch', int, 0),  # 400
            SSL_Argument('--cossl_tfe_warm_epoch', int, 1),  # 10
            SSL_Argument('--cossl_tfe_warm_lr', float, 0.02),
            SSL_Argument('--cossl_tfe_warm_ema_decay', float, 0.999),
            SSL_Argument('--cossl_tfe_warm_wd', float, 5e-4),
            SSL_Argument('--cossl_tfe_warm_bs', int, 64),
        ]

