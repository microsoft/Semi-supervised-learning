# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .hook import Hook


class ParamUpdateHook(Hook):
    """
    Parameter Update Hook

    necessary for update the model parameters
    """
    
    def before_train_step(self, algorithm):
        if hasattr(algorithm, 'start_run'):
            torch.cuda.synchronize()
            algorithm.start_run.record()

    # call after each train_step to update parameters
    def after_train_step(self, algorithm):
        loss = algorithm.out_dict['loss']
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()

        if algorithm.scheduler is not None:
            algorithm.scheduler.step()
        algorithm.model.zero_grad()

        if hasattr(algorithm, 'end_run'):
            algorithm.end_run.record()
            torch.cuda.synchronize()
            algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.

