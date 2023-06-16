# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_ssl_data
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation

mean, std = {}, {}
mean['svhn'] = [0.4380, 0.4440, 0.4730]
std['svhn'] = [0.1751, 0.1771, 0.1744]
img_size = 32

def get_transform(mean, std, crop_size, train=True, crop_ratio=0.95):
    img_size = int(crop_size / crop_ratio)

    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def get_svhn(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    img_size = int(math.floor(crop_size / crop_ratio))

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        # RandomResizedCropAndInterpolation(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        # RandomResizedCropAndInterpolation((crop_size, crop_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])


    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset_base = dset(data_dir, split='train', download=True)
    data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
    dset_extra = dset(data_dir, split='extra', download=True)
    data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
    data = np.concatenate([data_b, data_e])
    targets = np.concatenate([targets_b, targets_e])
    del data_b, data_e
    del targets_b, targets_e
    

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
                
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

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes,
                           transform_weak, False, transform_strong, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes,
                            transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, split='test', download=True)
    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
