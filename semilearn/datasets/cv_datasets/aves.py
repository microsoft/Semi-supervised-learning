# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import math
from torchvision.datasets import folder as dataset_parser
from torchvision.transforms import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from .datasetbase import BasicDataset


def get_semi_aves(args, alg, dataset, train_split='l_train_val', ulb_split='u_train_in', data_dir='./data'):
    assert train_split in ['l_train', 'l_train_val']

    data_dir = os.path.join(data_dir, 'semi_fgvc')

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    # NOTE this dataset is inherently imbalanced with unknown distribution
    train_labeled_dataset = iNatDataset(alg, data_dir, train_split, dataset, transform=transform_weak, transform_strong=transform_strong)
    train_unlabeled_dataset = iNatDataset(alg, data_dir, ulb_split, dataset, is_ulb=True, transform=transform_weak, transform_strong=transform_strong)
    test_dataset = iNatDataset(alg, data_dir, 'test', dataset, transform=transform_val)

    num_data_per_cls = [0] * train_labeled_dataset.num_classes
    for l in train_labeled_dataset.targets:
        num_data_per_cls[l] += 1

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def make_dataset(dataset_root, split, task='All', pl_list=None):
    split_file_path = os.path.join(dataset_root, task, split + '.txt')

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    if task == 'semi_fungi':
        img = [x.strip('\n').rsplit('.JPG ') for x in img]
    # elif task[:9] == 'semi_aves':
    else:
        img = [x.strip('\n').rsplit() for x in img]

    ## Use PL + l_train
    if pl_list is not None:
        if task == 'semi_fungi':
            pl_list = [x.strip('\n').rsplit('.JPG ') for x in pl_list]
        # elif task[:9] == 'semi_aves':
        else:
            pl_list = [x.strip('\n').rsplit() for x in pl_list]
        img += pl_list

    for idx, x in enumerate(img):
        if task == 'semi_fungi':
            img[idx][0] = os.path.join(dataset_root, x[0] + '.JPG')
        else:
            img[idx][0] = os.path.join(dataset_root, x[0])
        img[idx][1] = int(x[1])

    classes = [x[1] for x in img]
    num_classes = len(set(classes))
    print('# images in {}: {}'.format(split, len(img)))
    return img, num_classes, classes


class iNatDataset(BasicDataset):
    def __init__(self, alg, dataset_root, split, task='All', transform=None, transform_strong=None,
                 loader=dataset_parser.default_loader, pl_list=None, is_ulb=False):

        self.alg = alg
        self.is_ulb = is_ulb
        self.loader = loader
        self.dataset_root = dataset_root
        self.task = task

        self.samples, self.num_classes, self.targets = make_dataset(self.dataset_root, split, self.task, pl_list=pl_list)

        self.transform = transform
        self.strong_transform = transform_strong
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"

        self.data = []
        for i in range(len(self.samples)):
            self.data.append(self.samples[i][0])
    
    def __sample__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        return img, target 


    def __len__(self):
        return len(self.data)