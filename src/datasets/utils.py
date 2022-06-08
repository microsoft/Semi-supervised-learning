# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import numpy as np
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from src.datasets.samplers.sampler import DistributedSampler
from io import BytesIO

# TODO: better way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


name2sampler = {'RandomSampler': DistributedSampler}


def split_ssl_data(args, data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, = sample_labeled_data(args, data, target, num_labels, num_classes, index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def sample_labeled_data(args, data, target,
                        num_labels, num_classes,
                        index=None, name=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    # dump_path = os.path.join(args.save_dir, args.save_name, 'sampled_label_idx.npy')
    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f'labels{args.num_labels}_seed{args.seed}_idx.npy')

    if os.path.exists(dump_path):
        lb_idx = np.load(dump_path)
        lb_data = data[lb_idx]
        lbs = target[lb_idx]
        return lb_data, lbs, lb_idx
    

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    np.save(dump_path, np.array(lb_idx))

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__
                               if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


def get_collactor(args, net):
    if net == 'bert_base_uncased':
        from src.datasets.collactors import get_bert_base_uncased_collacor
        collact_fn = get_bert_base_uncased_collacor(args.max_length)
    elif net == 'bert_base_cased':
        from src.datasets.collactors import get_bert_base_cased_collacor
        collact_fn = get_bert_base_cased_collacor(args.max_length)
    elif net == 'wave2vecv2_base':
        from src.datasets.collactors import get_wave2vecv2_base_collacor
        collact_fn = get_wave2vecv2_base_collacor(args.max_length_seconds, args.sample_rate)
    elif net == 'hubert_base':
        from src.datasets.collactors import get_hubert_base_collacor
        collact_fn = get_hubert_base_collacor(args.max_length_seconds, args.sample_rate)
    else:
        collact_fn = None
    return collact_fn



def get_data_loader(args,
                    dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler='RandomSampler',
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None
    if num_epochs is None:
        num_epochs = args.epoch
    if num_iters is None:
        num_iters = args.num_train_iter
        
    collact_fn = get_collactor(args, args.net)

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collact_fn,
                          num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)

    if isinstance(data_sampler, str):
        data_sampler = name2sampler[data_sampler]

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0

        per_epoch_steps = num_iters // num_epochs

        num_samples = per_epoch_steps * batch_size * num_replicas

        return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collact_fn,
                          pin_memory=pin_memory, sampler=data_sampler(dset, num_replicas=num_replicas, rank=rank, num_samples=num_samples),
                          generator=generator, drop_last=drop_last)

    elif isinstance(data_sampler, torch.utils.data.Sampler):
        return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          collate_fn=collact_fn, pin_memory=pin_memory, sampler=data_sampler,
                          generator=generator, drop_last=drop_last)

    else:
        raise Exception(f"unknown data sampler {data_sampler}.")



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
