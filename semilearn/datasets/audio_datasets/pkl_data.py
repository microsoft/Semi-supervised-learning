# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import pickle
import numpy as np
from glob import glob
from semilearn.datasets.utils import split_ssl_data, bytes_to_array
from .datasetbase import BasicDataset



def get_pkl_dset(args, alg='fixmatch', dataset='esc50', num_labels=40, num_classes=20, data_dir='./data', include_lb_to_ulb=True, onehot=False):
    """
    get_ssl_dset split training samples into labeled and unlabeled samples.
    The labeled data is balanced samples over classes.
    
    Args:
        num_labels: number of labeled data.
        index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
        include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
        strong_transform: list of strong transform (RandAugment in FixMatch)
        onehot: If True, the target is converted into onehot vector.
        
    Returns:
        BasicDataset (for labeled data), BasicDataset (for unlabeled data)
    """
    data_dir = os.path.join(data_dir, dataset)

    # Supervised top line using all data as labeled data.
    if dataset == 'superbsi':
        all_train_files = sorted(glob(os.path.join(data_dir, 'train_*.pkl')))
        train_wav_list = []
        train_label_list = []
        for train_file in all_train_files:
            with open(train_file, 'rb') as f:
                train_data = pickle.load(f)
            for idx in train_data:
                train_wav_list.append(bytes_to_array(train_data[idx]['wav']))
                train_label_list.append(int(train_data[idx]['label']))
    else:
        with open(os.path.join(data_dir, 'train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        train_wav_list = []
        train_label_list = []
        for idx in train_data:
            train_wav_list.append(bytes_to_array(train_data[idx]['wav']))
            train_label_list.append(int(train_data[idx]['label']))

    with open(os.path.join(data_dir, 'dev.pkl'), 'rb') as f:
        dev_data = pickle.load(f)
    dev_wav_list = []
    dev_label_list = []
    for idx in dev_data:
        dev_wav_list.append(bytes_to_array(dev_data[idx]['wav']))
        dev_label_list.append(int(dev_data[idx]['label']))

    with open(os.path.join(data_dir, 'test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    test_wav_list = []
    test_label_list = []
    for idx in test_data:
        test_wav_list.append(bytes_to_array(test_data[idx]['wav']))
        test_label_list.append(int(test_data[idx]['label']))

    dev_dset = BasicDataset(alg=alg, data=dev_wav_list, targets=dev_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=False)
    test_dset = BasicDataset(alg=alg, data=test_wav_list, targets=test_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=False)
    if alg == 'fullysupervised':
        lb_dset = BasicDataset(alg=alg, data=train_wav_list, targets=train_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=True)
        return lb_dset, None, dev_dset, test_dset
    
    if dataset == 'fsdnoisy':
        # TODO: take care of this for imbalanced setting
        ulb_wav_list = []
        ulb_label_list = []
        with open(os.path.join(data_dir, 'ulb_train.pkl'), 'rb') as f:
            ulb_train_data = pickle.load(f)
        for idx in ulb_train_data:
            ulb_wav_list.append(bytes_to_array(ulb_train_data[idx]["wav"]))
            ulb_label_list.append(int(ulb_train_data[idx]["label"]))
        lb_wav_list, lb_label_list = train_wav_list, train_label_list
    else: 
        lb_wav_list, lb_label_list, ulb_wav_list, ulb_label_list = split_ssl_data(args, train_wav_list, train_label_list, num_classes, 
                                                                                  lb_num_labels=num_labels,
                                                                                  ulb_num_labels=args.ulb_num_labels,
                                                                                  lb_imbalance_ratio=args.lb_imb_ratio,
                                                                                  ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                                  include_lb_to_ulb=include_lb_to_ulb)

    # output the distribution of labeled data for remixmatch
    count = [0 for _ in range(num_classes)]
    for c in train_label_list:
        count[c] += 1
    dist = np.array(count, dtype=float)
    dist = dist / dist.sum()
    dist = dist.tolist()
    out = {"distribution": dist}
    output_file = r"./data_statistics/"
    output_path = output_file + str(dataset) + '_' + str(num_labels) + '.json'
    if not os.path.exists(output_file):
        os.makedirs(output_file, exist_ok=True)
    with open(output_path, 'w') as w:
        json.dump(out, w)
            
    lb_dset = BasicDataset(alg=alg, data=lb_wav_list, targets=lb_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=True)
    ulb_dset = BasicDataset(alg=alg, data=ulb_wav_list, targets=ulb_label_list, num_classes=num_classes, is_ulb=True, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=True)
    return lb_dset, ulb_dset, dev_dset, test_dset