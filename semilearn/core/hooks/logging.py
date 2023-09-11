# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref:https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/logger/base.py

from .hook import Hook


class LoggingHook(Hook):
    """
    Logging Hook for print information and log into tensorboard
    """

    def after_train_step(self, algorithm):
        """must be called after evaluation"""
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            if not algorithm.distributed or (
                algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0
            ):
                print_text = (
                    f"{algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, "
                )
                for i, (key, item) in enumerate(algorithm.log_dict.items()):
                    print_text += "{:s}: {:.4f}".format(key, item)
                    if i != len(algorithm.log_dict) - 1:
                        print_text += ", "
                    else:
                        print_text += " "

                print_text += "BEST_EVAL_ACC: {:.4f}, at {:d} iters".format(
                    algorithm.best_eval_acc, algorithm.best_it + 1
                )
                algorithm.print_fn(print_text)

            if algorithm.tb_log is not None:
                algorithm.tb_log.update(algorithm.log_dict, algorithm.it)

        elif self.every_n_iters(algorithm, algorithm.num_log_iter):
            if not algorithm.distributed or (
                algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0
            ):
                print_text = (
                    f"{algorithm.it + 1} iteration USE_EMA: {algorithm.ema_m != 0}, "
                )
                for i, (key, item) in enumerate(algorithm.log_dict.items()):
                    print_text += "{:s}: {:.4f}".format(key, item)
                    if i != len(algorithm.log_dict) - 1:
                        print_text += ", "
                    else:
                        print_text += " "
                algorithm.print_fn(print_text)

            if algorithm.tb_log is not None:
                algorithm.tb_log.update(algorithm.log_dict, algorithm.it)
