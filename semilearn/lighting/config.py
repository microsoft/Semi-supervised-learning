# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from semilearn.algorithms import name2alg
from semilearn.imb_algorithms import name2imbalg
from semilearn.algorithms.utils import str2bool
from semilearn.core.utils import over_write_args_from_dict, get_port


def get_config(config):

    parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB semilearn package)')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb to plot and save curves')
    parser.add_argument('--use_aim', action='store_true', help='Use aim to plot and save curves')

    '''
    Training Configuration of FixMatch
    '''
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=20,
                        help='total number of training iterations')
    parser.add_argument('--num_warmup_iter', type=int, default=0,
                        help='cosine linear warmup iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10,
                        help='evaluation frequency')
    parser.add_argument('--num_log_iter', type=int, default=5,
                        help='logging frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=400)
    parser.add_argument('-bsz', '--batch_size', type=int, default=8)
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeled data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=1.0, help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--use_pretrain', default=False, type=str2bool)
    parser.add_argument('--pretrain_path', default='', type=str)

    '''
    Algorithms Configurations
    '''  

    ## core algorithm setting
    parser.add_argument('-alg', '--algorithm', type=str, default='fixmatch', help='ssl algorithm')
    parser.add_argument('--use_cat', type=str2bool, default=True, help='use cat operation in algorithms')
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument('-imb_alg', '--imb_algorithm', type=str, default=None, help='imbalance ssl algorithm')

    '''
    Data Configurations
    '''

    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default='True', help='flag of including labeled data into unlabeled data, default to True')

    ## imbalanced setting arguments
    parser.add_argument('--lb_imb_ratio', type=int, default=1, help="imbalance ratio of labeled data, default to 1")
    parser.add_argument('--ulb_imb_ratio', type=int, default=1, help="imbalance ratio of unlabeled data, default to 1")
    parser.add_argument('--ulb_num_labels', type=int, default=None, help="number of labels for unlabeled data, used for determining the maximum number of labels in imbalanced setting")

    ## cv dataset arguments
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)

    ## nlp dataset arguments 
    parser.add_argument('--max_length', type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    '''
    multi-GPUs & Distributed Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    # add algorithm specific parameters
    args = parser.parse_args("")
    over_write_args_from_dict(args, config)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)


    # add imbalanced algorithm specific parameters
    args = parser.parse_args("")
    over_write_args_from_dict(args, config)
    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)

    args = parser.parse_args("")    
    over_write_args_from_dict(args, config)


    if args.save_name is None:
        args.save_name = args.algorithm + '_' + args.dataset

    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    return args