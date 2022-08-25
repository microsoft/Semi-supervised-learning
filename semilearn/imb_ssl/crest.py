# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch

from semilearn.datasets import DistributedSampler
from semilearn.algorithms.utils import SSL_Argument, EMA
from semilearn.algorithms.algorithmbase import AlgorithmBase


class CReST(AlgorithmBase):
    def __init__(self, *args, **kwargs):
        super(CReST, self).__init__(*args, **kwargs)
    
    def init(self, num_gens=6, dist_align_t=0.5, sampling_alpha=3, *args, **kwargs):
        self.num_gens = num_gens
        self.dist_align_t = dist_align_t
        self.sampling_alpha = sampling_alpha
        self.start_gen = 0
        super().init(*args, **kwargs)

        # TODO: set dist alignment

    @staticmethod
    def get_argument():
        argument = super().get_argument()
        argument.extend([
            SSL_Argument('--num_gens', int, 6),
            SSL_Argument('--dsit_align_t', float, 0.5),
            SSL_Argument('--sampling_alpha', float, 3),
        ])
        return argument


    def re_init(self):
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.ema = None
        
        # build dataset
        # TODO, reset dataset 
        self.dataset_dict = self.set_dataset()

        # cv, nlp, speech builder different arguments
        self.model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
        self.ema_model = self.set_ema_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build data loader
        self.loader_dict = self.set_data_loader()

    def train(self):

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if self.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        start_batch.record()


        for gen in range(self.start_gen, self.num_gens):
            self.gen = gen
            cur = gen / ( self.num_gens - 1)
            cur_dist_align_t = (1.0 - cur) * 1.0 + cur * self.dist_align_t

            self.re_init()

            for epoch in range(self.epochs):
                # TODO: move this part to before train epoch
                self.epoch = epoch
                
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                    self.loader_dict['train_lb'].sampler.set_epoch(epoch)
                if isinstance(self.loader_dict['train_ulb'].sampler, DistributedSampler):
                    self.loader_dict['train_ulb'].sampler.set_epoch(epoch)

                # for (idx_lb, x_lb, y_lb), (idx_ulb, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                #                                                              self.loader_dict['train_ulb']):
                for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                            self.loader_dict['train_ulb']):
                    # prevent the training iterations exceed args.num_train_iter
                    if self.it >= self.num_train_iter:
                        break

                    end_batch.record()
                    torch.cuda.synchronize()
                    start_run.record()

                    # self.tb_dict = self.train_step(**self.process_batch(idx_lb=idx_lb, x_lb=x_lb, y_lb=y_lb, idx_ulb=idx_ulb, x_ulb_w=x_ulb_w, x_ulb_s=x_ulb_s))
                    self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))

                    end_run.record()
                    torch.cuda.synchronize()

                    # tensorboard_dict update
                    self.tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                    self.tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                    self.tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                    # post processing for saving model and print logs
                    self.after_train_step()
                    start_batch.record()

            eval_dict = {'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it}
            if 'test' in self.loader_dict:
                # load the best model and evaluate on test dataset
                best_model_path = os.path.join(self.args.save_dir, self.args.save_name, 'model_best.pth')
                self.load_model(best_model_path)
                test_dict = self.evaluate('test')
                eval_dict['test/best_acc'] = test_dict['test/top-1-acc']
        
        return eval_dict


        

        
def create_crest(CoreAlgorithm, *args, **kwargs):
    class DummyClass(CReST, CoreAlgorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    return DummyClass(*args, **kwargs)

# if __name__ == '__main__':
#     from semilearn.algorithms.fixmatch import FixMatch
#     alg = crest(FixMatch)
#     print(alg)