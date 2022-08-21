# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data'):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_labels, num_classes, None, True)
    if alg == 'fullysupervised':
        if len(ulb_data) == len(data):
            lb_data = ulb_data 
            lb_targets = ulb_targets
        else:
            lb_data = np.concatenate([lb_data, ulb_data], axis=0)
            lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        count[c] += 1
    dist = np.array(count, dtype=float)
    dist = dist / dist.sum()
    dist = dist.tolist()
    out = {"distribution": dist}
    output_file = r"./data_statistics/"
    output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    if not os.path.exists(output_file):
        os.makedirs(output_file, exist_ok=True)
    with open(output_path, 'w') as w:
        json.dump(out, w)

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
