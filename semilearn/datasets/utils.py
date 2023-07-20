# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import numpy as np
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from io import BytesIO

# TODO: better way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def split_ssl_data(args, data, targets, num_classes,
                   lb_num_labels, ulb_num_labels=None,
                   lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True, load_exist=True):
    """
    data & target is splitted into labeled and unlabeled data.
    
    Args
        data: data to be split to labeled and unlabeled 
        targets: targets to be split to labeled and unlabeled 
        num_classes: number of total classes
        lb_num_labels: number of labeled samples. 
                       If lb_imbalance_ratio is 1.0, lb_num_labels denotes total number of samples.
                       Otherwise it denotes the number of samples in head class.
        ulb_num_labels: similar to lb_num_labels but for unlabeled data.
                        default to None, denoting use all remaining data except for labeled data as unlabeled set
        lb_imbalance_ratio: imbalance ratio for labeled data
        ulb_imbalance_ratio: imbalance ratio for unlabeled data
        lb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeled data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes, 
                                                    lb_num_labels, ulb_num_labels,
                                                    lb_imbalance_ratio, ulb_imbalance_ratio, load_exist=False)
    
    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    
    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]


def sample_labeled_unlabeled_data(args, data, target, num_classes,
                                  lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                                  load_exist=True):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_{args.lb_imb_ratio}_seed{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_{args.ulb_imb_ratio}_seed{args.seed}_idx.npy')

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx 

    
    # get samples per class
    if lb_imbalance_ratio == 1.0:
        # balanced setting, lb_num_labels is total number of labels for labeled data
        assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
        lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
    else:
        # imbalanced setting, lb_num_labels is the maximum number of labels for class 1
        lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio)


    if ulb_imbalance_ratio == 1.0:
        # balanced setting
        if ulb_num_labels is None or ulb_num_labels == 'None':
            ulb_samples_per_class = None
        else:
            assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be dividable by num_classes in balanced setting"
            ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes
    else:
        # imbalanced setting
        assert ulb_num_labels is not None, "ulb_num_labels must be set set in imbalanced setting"
        ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio)

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        if ulb_samples_per_class is None:
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)
    
    return lb_idx, ulb_idx


def make_imbalance_data(max_num_labels, num_classes, gamma):
    """
    calculate samplers per class for imbalanced data
    """
    mu = np.power(1 / abs(gamma), 1 / (num_classes - 1))
    samples_per_class = []
    for c in range(num_classes):
        if c == (num_classes - 1):
            samples_per_class.append(int(max_num_labels / abs(gamma)))
        else:
            samples_per_class.append(int(max_num_labels * np.power(mu, c)))
    if gamma < 0:
        samples_per_class = samples_per_class[::-1]
    return samples_per_class


def get_collactor(args, net):
    if net == 'bert_base_uncased':
        from semilearn.datasets.collactors import get_bert_base_uncased_collactor
        collact_fn = get_bert_base_uncased_collactor(args.max_length)
    elif net == 'bert_base_cased':
        from semilearn.datasets.collactors import get_bert_base_cased_collactor
        collact_fn = get_bert_base_cased_collactor(args.max_length)
    elif net == 'wave2vecv2_base':
        from semilearn.datasets.collactors import get_wave2vecv2_base_collactor
        collact_fn = get_wave2vecv2_base_collactor(args.max_length_seconds, args.sample_rate)
    elif net == 'hubert_base':
        from semilearn.datasets.collactors import get_hubert_base_collactor
        collact_fn = get_hubert_base_collactor(args.max_length_seconds, args.sample_rate)
    else:
        collact_fn = None
    return collact_fn



def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]