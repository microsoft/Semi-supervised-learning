# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.algorithms.algorithmbase import AlgorithmBase
from src.datasets.samplers.sampler import DistributedSampler
from src.algorithms.utils import ce_loss, EMA, SSL_Argument, str2bool
from src.datasets.utils import get_data_loader
from PIL import Image


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return img.rot90(1, [1, 2])
    elif rot == 180:  # 90 degrees rotation
        return img.rot90(2, [1, 2])
    elif rot == 270:  # 270 degrees rotation / or -90
        return img.rot90(1, [2, 1])
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class RotNet(torch.utils.data.Dataset):
    '''
    Dataloader for RotNet
    the image first goes through data augmentation, and then rotate 4 times
    the output is 4 rotated views of the augmented image,
    the corresponding labels are 0 1 2 3
    '''
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.data[index]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, str):
            img = Image.open(img)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return img, rotate_img(img, 90), rotate_img(img, 180), rotate_img(img, 270), rotation_labels

    def __len__(self):
        return len(self.data)


class CRMatch_Net(nn.Module):
    def __init__(self, base, args, use_rot=True):
        super(CRMatch_Net, self).__init__()
        self.backbone = base
        self.use_rot = use_rot
        self.feat_planes = base.num_features
        self.args = args

        if self.use_rot:
            self.rot_classifier = nn.Sequential(
                nn.Linear(self.feat_planes, self.feat_planes),
                nn.ReLU(inplace=False),
                nn.Linear(self.feat_planes, 4)
            )
        if 'wrn' in args.net or 'resnet' in args.net:
            if args.dataset == 'stl10':
                feat_map_size = 6 * 6 * self.feat_planes
            elif args.dataset == 'imagenet':
                feat_map_size = 7 * 7 * self.feat_planes
            else:
                feat_map_size = 8 * 8 * self.feat_planes
        elif 'vit' in args.net or 'bert' in args.net or 'wave2vec' in args.net:
            feat_map_size = self.backbone.num_features
        else:
            raise NotImplementedError
        self.ds_classifier = nn.Linear(feat_map_size, self.feat_planes, bias=True)

    def forward(self, x):
        feat_maps = self.backbone.extract(x)

        if 'wrn' in self.args.net or 'resnet' in self.args.net:
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
            feat_maps = torch.mean(feat_maps, dim=(2, 3))
        elif 'vit' in self.args.net:
            if self.backbone.global_pool:
                feat_maps = feat_maps[:, 1:].mean(dim=1) if self.backbone.global_pool == 'avg' else feat_maps[:, 0]
            feat_maps = self.backbone.fc_norm(feat_maps)
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
        elif 'bert' in self.args.net or 'wave2vec' in self.args.net:
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
        else:
            raise NotImplementedError
        logits = self.backbone(feat_maps, only_fc=True)
        # feat_flat = torch.mean(feat_maps, dim=(2, 3))
        # logits = self.backbone(feat_flat, only_fc=True)
        if self.use_rot:
            logits_rot = self.rot_classifier(feat_maps)
        else:
            logits_rot = None
        return logits, logits_rot, logits_ds


class CRMatch(AlgorithmBase):
    """
        CRMatch algorithm (https://arxiv.org/abs/2112.05825).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder,  tb_log, logger)
        # crmatch specificed arguments
        self.init(p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.lambda_rot = args.rot_loss_ratio
        self.use_rot = self.lambda_rot > 0
        self.model_backbone = self.model
        self.model = CRMatch_Net(self.model_backbone, args, use_rot=self.use_rot)
        self.ema_model = CRMatch_Net(self.ema_model, args, use_rot=self.use_rot)
        self.ema_model.load_state_dict(self.model.state_dict())
        

    def init(self, p_cutoff, hard_label=True):
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label


    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

        if self.use_rot:
            x_ulb_rot = deepcopy(self.loader_dict['train_ulb'].dataset.data)
            dataset_ulb_rot = RotNet(x_ulb_rot, transform=self.loader_dict['train_lb'].dataset.transform)
            self.loader_dict['train_ulb_rot'] = get_data_loader(self.args,
                                                                dataset_ulb_rot,
                                                                self.args.batch_size,
                                                                data_sampler=self.args.train_sampler,
                                                                num_iters=self.args.num_train_iter,
                                                                num_epochs=self.args.epoch,
                                                                num_workers=4 * self.args.num_workers,
                                                                distributed=self.args.distributed)
            self.loader_dict['train_ulb_rot_iter'] = iter(self.loader_dict['train_ulb_rot'])

    def train(self):
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
            if self.it > self.num_train_iter:
                break
                
            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)
            if isinstance(self.loader_dict['train_ulb'].sampler, DistributedSampler):
                self.loader_dict['train_ulb'].sampler.set_epoch(epoch)
            if 'train_ulb_rot' in self.loader_dict and isinstance(self.loader_dict['train_ulb_rot'].sampler, DistributedSampler):
                self.loader_dict['train_ulb_rot'].sampler.set_epoch(epoch)
            
            for data_lb, data_ulb in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                if self.use_rot:
                    try:
                        img, img90, img180, img270, rot_v = next(self.loader_dict['train_ulb_rot_iter'])
                    except:
                        self.loader_dict['train_ulb_rot_iter'] = iter(self.loader_dict['train_ulb_rot'])
                        img, img90, img180, img270, rot_v = next(self.loader_dict['train_ulb_rot_iter'])
                    x_ulb_rot = torch.cat((img, img90, img180, img270), dim=0).contiguous()
                    rot_v = rot_v.transpose(1, 0).contiguous().view(-1)
                else:
                    x_ulb_rot = None
                    rot_v = None

                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb, x_ulb_rot=x_ulb_rot, rot_v=rot_v))

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                self.tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                self.tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                self.tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                self.after_train_step()
                start_batch.record()

        eval_dict = self.evaluate()
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        return eval_dict


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, x_ulb_rot=None, rot_v=None):
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                if self.use_rot:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s, x_ulb_rot), dim=0).contiguous()
                else:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s), dim=0).contiguous()
                logits, logits_rot, logits_ds = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:num_lb + 2 * num_ulb].chunk(2)
                logits_ds_w, logits_ds_s = logits_ds[num_lb:num_lb + 2 * num_ulb].chunk(2)
            else:
                logits_x_lb, _, _ = self.model(x_lb)
                logits_x_ulb_s, _, logits_ds_s = self.model(x_ulb_s)
                with torch.no_grad():
                    logits_x_ulb_w, _, logits_ds_w = self.model(x_ulb_w)
               

            with torch.no_grad():
                pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
                max_probs, y_ulb = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.p_cutoff).float()

            Lx = ce_loss(logits_x_lb, y_lb, reduction='mean')
            Lu = (ce_loss(logits_x_ulb_s, y_ulb, reduction='none') * mask).mean()
            Ld = F.cosine_embedding_loss(logits_ds_s, logits_ds_w, -torch.ones(logits_ds_s.size(0)).float().cuda(self.gpu), reduction='none')
            Ld = (Ld * mask).mean()

            total_loss = Lx + Lu + Ld

            if self.use_rot:
                if self.use_cat:
                    logits_rot = logits_rot[num_lb + 2 * num_ulb:]
                else:
                    _, logits_rot, _ = self.model(x_ulb_rot)
                Lrot = ce_loss(logits_rot, rot_v, reduction='mean')
                total_loss += Lrot

        # parameter updates
        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = Lx.item()
        tb_dict['train/unsup_loss'] = Lu.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--rot_loss_ratio', float, 1.0, 'weight for rot loss, set to 0 for nlp and speech'),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
