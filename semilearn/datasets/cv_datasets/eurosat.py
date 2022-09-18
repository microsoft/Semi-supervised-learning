# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
EuroSat has 27000 images for 10 classes 'AnnualCrop', 'Forest', 'HerbaceousVegetation',
'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'.

I only use the RGB Sentinel-2 satellite images.

VTAB just take the first 60% images from each class as training images, then the next
20% as val images, and then the last 20% as test images.

Original paper (https://arxiv.org/pdf/1709.00029.pdf) does not use a val set, it simply
split the dataset into a training and a test set with different ratios (from 10/90 to 90/10).
Here I define 3 hyper-parameters: TRAIN_SPLIT_PERCENT, VALIDATION_SPLIT_PERCENT, TEST_SPLIT_PERCENT.

Each image is of size 64x64x3.

Note that for now, I only stick to 80/20 split between training and test images, and do not
have a val set. Because each class has different number of images. And I ensured that the
split is applied class-wise. So there is a small imbalance among classes

"""

import os
import numpy as np
import copy
import math 
import random
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.utils import split_ssl_data
from .datasetbase import BasicDataset


dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

num_classes = 10


def get_eurosat(args, alg, dataset, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    img_size = int(math.floor(crop_size / crop_ratio))

    # transform_train_eurosat = transforms.Compose([
    #         transforms.Resize(int(math.floor(img_size / crop_ratio))),
    #         transforms.RandomHorizontalFlip(),
            
    #         transforms.RandomCrop((img_size, img_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize()
    #     ])

    # transform_val_eurosat = transforms.Compose([
    #     transforms.Resize(int(math.floor(img_size / crop_ratio))),
    #     transforms.CenterCrop((img_size, img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(dataset_mean, dataset_std)
    # ])

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        # RandomResizedCropAndInterpolation(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        # RandomResizedCropAndInterpolation((crop_size, crop_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])


    data_dir = os.path.join(data_dir, dataset.lower())
    base_dataset = EuroSat(alg, data_dir, split="trainval")


    # num_classes = 10
    n_labeled_per_class = int(num_labels // num_classes)

    train_targets = base_dataset.targets
    train_ids = base_dataset.idx_list
    assert len(train_targets) == len(train_ids), "EuroSat dataset has an error!!!"

    # shuffle the dataset
    shuffle_index = list(range(len(train_ids)))
    np.random.shuffle(shuffle_index)
    total_targets = train_targets[shuffle_index]
    total_idxs = train_ids[shuffle_index]

    train_labeled_idxs, _, train_unlabeled_idxs, _ = split_ssl_data(args, total_idxs, total_targets, num_classes, 
                                                                    lb_num_labels=num_labels,
                                                                    ulb_num_labels=args.ulb_num_labels,
                                                                    lb_imbalance_ratio=args.lb_imb_ratio,
                                                                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                    include_lb_to_ulb=include_lb_to_ulb)
    # construct datasets for training and testing
    if alg == 'fullysupervised':
        if len(train_unlabeled_idxs) == len(total_idxs):
            train_labeled_idxs = train_unlabeled_idxs 
        else:
            train_labeled_idxs = np.concatenate([train_labeled_idxs, train_unlabeled_idxs])
    
    train_labeled_dataset = EuroSat(alg, data_dir, split="trainval", idx_list=train_labeled_idxs, transform=transform_weak)
    train_unlabeled_dataset = EuroSat(alg, data_dir, split="trainval", is_ulb=True, idx_list=train_unlabeled_idxs, transform=transform_weak, transform_strong=transform_strong)
    val_dataset = EuroSat(alg, data_dir, split="test", transform=transform_val)

    print(f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} "
          f"#Val: {len(val_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


def balanced_selection(total_data, total_targets, num_classes, per_class_data):
    select_index_set = np.zeros(num_classes * per_class_data, dtype=int) - 1
    label_counter = [0] * num_classes
    j = 0
    for i, label in enumerate(total_targets):
        if label_counter[label] != per_class_data:
            label_counter[label] += 1
            select_index_set[j] = i
            j += 1
        if label_counter == [per_class_data] * num_classes:
            break
    unselected_index_set = np.ones(total_targets.shape).astype(bool)
    unselected_index_set[select_index_set] = 0
    unselected_index_set, = np.where(unselected_index_set)

    selected_data = total_data[select_index_set]
    selected_targets = total_targets[select_index_set]
    unselected_data = total_data[unselected_index_set]
    unselected_targets = total_targets[unselected_index_set]
    return selected_data, selected_targets, unselected_data, unselected_targets


class EuroSat(ImageFolder, BasicDataset):

    TRAIN_SPLIT_PERCENT = 0.60
    VALIDATION_SPLIT_PERCENT = 0.20
    TEST_SPLIT_PERCENT = 0.20
    # todo: implement _check_integrity method here!

    def __init__(self, alg, root, split, is_ulb=False, idx_list=None, transform=None, target_transform=None, transform_strong=None):
        """see comments at the beginning of the script"""
        super(EuroSat, self).__init__(root, transform=transform, target_transform=target_transform)

        self.is_ulb = is_ulb
        self.alg = alg
        self.strong_transform = transform_strong
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"

        unique, counts = np.unique(self.targets, return_counts=True)
        self.num_imgs_per_class = dict(zip(unique, counts))  # dont use os.listdir! due to the order!!!

        if split == 'trainval':
            self.idx_list = []
            tmp = 0
            for cls, num_imgs in self.num_imgs_per_class.items():
                self.idx_list += list(range(tmp,
                                            int((self.TRAIN_SPLIT_PERCENT + self.VALIDATION_SPLIT_PERCENT) * num_imgs) + tmp))
                tmp += num_imgs
        elif split == 'test':
            self.idx_list = []
            tmp = 0
            for cls, num_imgs in self.num_imgs_per_class.items():
                self.idx_list += list(range(int((self.TRAIN_SPLIT_PERCENT + self.VALIDATION_SPLIT_PERCENT) * num_imgs) + tmp,
                                            tmp + num_imgs))
                tmp += num_imgs
        else:
            raise Exception('unknown split parameter for EuroSat!!!')

        self.idx_list = np.asarray(self.idx_list)
        self.targets = np.asarray(self.targets)[self.idx_list]

        if idx_list is not None:
            self.idx_list = idx_list
        
        self.data = []
        for i in range(len(self.samples)):
            self.data.append(self.samples[i][0])
    
    def __sample__(self, index):
        idx = self.idx_list[index]
        path, target = self.samples[idx]
        img = self.loader(path)
        return img, target
    
    def __getitem__(self, index):
        return BasicDataset.__getitem__(self, index)

    def __len__(self):
        return len(self.idx_list)


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.datasets import ImageFolder
    root = '/BS/yfan/nobackup/VTAB/eurosat/2750'

    def is_grey_scale(img):
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i, j))
                if r != g != b:
                    return False
        return True

    # dataset = ImageFolder(root)
    # print(dataset.targets)
    # print(len(dataset.samples))
    # print(dataset.samples[6000])

    # a = EuroSat(root, split='trainval')
    # b = EuroSat(root, split='test')
    # trainval = a.idx_list
    # test = b.idx_list
    #
    # c = trainval + test
    # print(len(c))
    # print(len(np.unique(c)))

    # a = EuroSat(root, split='trainval')
    # print(len(a.targets))
    # # import collections
    # unique, counts = np.unique(a.targets, return_counts=True)
    # print(dict(zip(unique, counts)))

    # train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = get_datasets(root, 5)
    # c = [0] * 10
    # for img, label in test_dataset:
    #     c[label] += 1
    # print(c)
    # c = 0
    # for img, label in test_dataset:
    #     if label == 2:
    #         print(label)
    #         c += 1
    #         plt.imshow(img)
    #         plt.show()
    #     if c == 10:
    #         break


