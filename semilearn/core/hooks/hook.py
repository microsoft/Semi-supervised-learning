# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/hook.py


class Hook:
    stages = ('before_run', 
              'before_train_epoch', 'before_train_step', 'after_train_step', 'after_train_epoch',
              'after_run')

    def before_train_epoch(self, algorithm):
        pass

    def after_train_epoch(self, algorithm):
        pass

    def before_train_step(self, algorithm):
        pass

    def after_train_step(self, algorithm):
        pass
    
    def before_run(self, algorithm):
        pass

    def after_run(self, algorithm):
        pass

    def every_n_epochs(self, algorithm, n):
        return (algorithm.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, algorithm, n):
        return (algorithm.it + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, algorithm):
        return algorithm.it + 1 % len(algorithm.data_loader['train_lb']) == 0

    def is_last_epoch(self, algorithm):
        return algorithm.epoch + 1 == algorithm.epochs

    def is_last_iter(self, algorithm):
        return algorithm.it + 1 == algorithm.num_train_iter