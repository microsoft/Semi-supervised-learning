# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref:https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/logger/base.py

from .hook import Hook

class LoggingHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    

    def after_train_step(self, algorithm):
        """must be called after evaluation"""
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.print_fn(f"{algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, {algorithm.tb_dict}, BEST_EVAL_ACC: {algorithm.best_eval_acc}, at {algorithm.best_it + 1} iters")
            
            if not algorithm.tb_log is None:
                algorithm.tb_log.update(algorithm.tb_dict, algorithm.it)
        
        elif self.every_n_iters(algorithm, algorithm.num_log_iter):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.print_fn(f"{algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, {algorithm.tb_dict}")