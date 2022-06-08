# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import os
import time
import random

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import logging
import ruamel.yaml as yaml



def over_write_args_from_dict(args, dict):
    """
    overwrite arguments acocrding to config file
    """
    for k in dict:
        setattr(args, k, dict[k])


def over_write_args_from_file(args, yml):
    """
    overwrite arguments acocrding to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def net_builder(net_name, from_name: bool):
    """
    built network according to network name
    return **class** of backbone network (not instance).

    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]
    else:
        import src.nets as net
        builder = getattr(net, net_name)
        return builder


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_dataset(args, algorithm, dataset, num_labels, num_classes, seed=0, data_dir='./data'):
    """
    create dataset

    Args
        args: argparse arguments
        algorithm: algrorithm name, used for specific return items in __getitem__ of datasets
        dataset: dataset name 
        num_labels: number of labeled data in dataset
        num_classes: number of classes
        seed: random seed
        data_dir: data folder
    """
    from src.datasets.cv_datasets import get_eurosat, get_medmnist, get_semi_aves, get_cifar, get_svhn, get_stl10, get_imagenet
    from src.datasets.nlp_datasets import get_json_dset
    from src.datasets.speech_datasets import get_pkl_dset

    if dataset == "eurosat":
        lb_dset, ulb_dset, eval_dset = get_eurosat(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir, seed=seed)
        test_dset = None
    elif dataset in ["tissuemnist"]:
        lb_dset, ulb_dset, eval_dset = get_medmnist(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir,  seed=seed)
        test_dset = None
    elif dataset == "semi_aves":
        lb_dset, ulb_dset, eval_dset = get_semi_aves(args, algorithm, dataset, train_split='l_train_val', data_dir=data_dir)
        test_dset = None
    elif dataset in ["cifar10", "cifar100"]:
        lb_dset, ulb_dset, eval_dset = get_cifar(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir)
        test_dset = None
    elif dataset == 'svhn':
        lb_dset, ulb_dset, eval_dset = get_svhn(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir)
        test_dset = None
    elif dataset == 'stl10':
        lb_dset, ulb_dset, eval_dset = get_stl10(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir)
        test_dset = None
    elif dataset == "imagenet":
        lb_dset, ulb_dset, eval_dset = get_imagenet(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir)
        test_dset = None
    # speech dataset
    elif dataset in ['esc50', 'fsdnoisy', 'gtzan', 'superbks', 'superbsi', 'urbansound8k']:
        lb_dset, ulb_dset, eval_dset, test_dset = get_pkl_dset(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir)
    elif dataset in ['aclImdb', 'ag_news', 'amazon_review', 'dbpedia', 'yahoo_answers', 'yelp_review']:
        lb_dset, ulb_dset, eval_dset, test_dset = get_json_dset(args, algorithm, dataset, num_labels, num_classes, data_dir=data_dir)
    else:
        raise NotImplementedError
        
    dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset}
    return dataset_dict


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name or 'bias' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)
    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_port():
    """
    find a free port to used for distributed learning
    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000, 30000)
    if tt not in procarr:
        return tt
    else:
        return get_port()


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
            

    def update(self, tb_dict, it, suffix=None, mode="train"):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value, it)