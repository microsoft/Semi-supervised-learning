# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import aim
from .hook import Hook


class AimHook(Hook):
    """
    Aim Hook
    """

    def __init__(self):
        super().__init__()
        self.log_key_list = ['train/sup_loss', 'train/unsup_loss', 'train/total_loss', 'train/util_ratio', 
                             'train/run_time', 'train/prefetch_time', 'lr',
                             'eval/top-1-acc', 'eval/precision', 'eval/recall', 'eval/F1']

    def before_run(self, algorithm):
        # initialize aim run
        name = algorithm.save_name
        project = algorithm.save_dir.split('/')[-1]
        self.run = aim.Run(experiment=name, repo='/mnt/default/projects/USB_formal_run/221124/aim_data')

        # set configuration
        self.run['hparams'] = algorithm.args.__dict__


        # set tag
        benchmark = f'benchmark: {project}'
        dataset = f'dataset: {algorithm.args.dataset}'
        data_setting = f'setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_{algorithm.args.lb_imb_ratio}_ulb{algorithm.args.ulb_num_labels}_{algorithm.args.ulb_imb_ratio}'
        alg = f'alg: {algorithm.args.algorithm}'
        imb_alg = f'imb_alg: {algorithm.args.imb_algorithm}'
        self.run.add_tag(benchmark)
        self.run.add_tag(dataset)
        self.run.add_tag(data_setting)
        self.run.add_tag(alg)
        self.run.add_tag(imb_alg)

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            for key, item in algorithm.log_dict.items():
                if key in self.log_key_list:
                    self.run.track(item, name=key, step=algorithm.it)
        
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            self.run.track(algorithm.best_eval_acc, name='eval/best-acc', step=algorithm.it)
