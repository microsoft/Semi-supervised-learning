
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
from semilearn.core.hooks import CheckpointHook, EvaluationHook, LoggingHook
from semilearn.algorithms.hooks import DistAlignEMAHook


class CReSTLoggingHook(LoggingHook):
    def after_train_step(self, algorithm):
        """must be called after evaluation"""
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.print_fn(f"{algorithm.gen + 1} generation {algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, {algorithm.tb_dict}, BEST_EVAL_ACC: {algorithm.best_eval_acc}, at {algorithm.best_it + 1} iters")
            
            if not algorithm.tb_log is None:
                algorithm.tb_log.update(algorithm.tb_dict, algorithm.it)
        
        elif self.every_n_iters(algorithm, algorithm.num_log_iter):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.print_fn(f"{algorithm.gen + 1} generation {algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, {algorithm.tb_dict}")


class CReSTCheckpointHook(CheckpointHook):
    def after_train_step(self, algorithm):
        # normal save for 
        super().after_train_step(algorithm)

        # save to generation folder
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name, f'gen_{algorithm.gen}')
            
            if (not algorithm.distributed) or \
               (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.save_model('latest_model.pth', save_path)

                if algorithm.it == algorithm.best_it:
                    algorithm.save_model('model_best.pth', save_path)


class CReSTEvaluationHook(EvaluationHook):
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.tb_dict.update(eval_dict)

            # update best metrics
            if algorithm.tb_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.tb_dict['eval/top-1-acc']
                algorithm.best_it = algorithm.it
                algorithm.best_gen = algorithm.gen
                algorithm.best_gen_val_acc =  algorithm.tb_dict['eval/top-1-acc']
    

    def after_run(self, algorithm):
        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it, 'eval/best_gen': algorithm.best_gen}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict



class ProgressiveDistAlignEMAHook(DistAlignEMAHook):
    """
    Progressive Distribution Alignment in CRest
    """
    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)

        # dist align
        p_target = self.p_target
        if algorithm.cur_dist_align_t != 1:
            p_target = p_target ** algorithm.cur_dist_align_t
            p_target = p_target / p_target.sum()
        probs_x_ulb_aligned = probs_x_ulb * (p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned