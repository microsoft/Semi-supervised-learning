# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .hook import Hook


class TimerHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    def before_run(self, algorithm):
        algorithm.start_batch = torch.cuda.Event(enable_timing=True)
        algorithm.end_batch = torch.cuda.Event(enable_timing=True)
        algorithm.start_run = torch.cuda.Event(enable_timing=True)
        algorithm.end_run = torch.cuda.Event(enable_timing=True)
        algorithm.start_batch.record()
    
    def before_train_step(self, algorithm):
        algorithm.end_batch.record()
        torch.cuda.synchronize()
        algorithm.start_run.record()
    
    def after_train_step(self, algorithm):
        algorithm.end_run.record()
        torch.cuda.synchronize()
        algorithm.tb_dict['lr'] = algorithm.optimizer.param_groups[-1]['lr']
        algorithm.tb_dict['train/prefecth_time'] = algorithm.start_batch.elapsed_time(algorithm.end_batch) / 1000.
        algorithm.tb_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.
        algorithm.start_batch.record()