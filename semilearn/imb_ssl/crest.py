# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import torch
import numpy as np

from semilearn.datasets import DistributedSampler
from semilearn.algorithms.utils import SSL_Argument, EMA
from semilearn.algorithms.algorithmbase import AlgorithmBase
from semilearn.datasets.utils import get_data_loader
from semilearn.utils import get_dataset


# TODO: add ImbAlgorithmBase
class CReST(AlgorithmBase):
    def __init__(self, *args, **kwargs):
        super(CReST, self).__init__(*args, **kwargs)
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio


    def init(self, num_gens=6, dist_align_t=0.5, sampling_alpha=3, *args, **kwargs):
        self.num_gens = num_gens
        self.dist_align_t = dist_align_t
        self.sampling_alpha = sampling_alpha
        self.start_gen = 0
        super().init(*args, **kwargs)

        # TODO: set dist alignment
    
    def get_split(self, lb_data, lb_targets, eval_ulb_data, eval_ulb_targets, pseudo_label_list=None):
        if pseudo_label_list is not None:
            data_picked = []
            targets_picked = []
            class_imb_ratio = self.lb_imb_ratio
            class_imb_ratio = 1. / class_imb_ratio
            mu = np.math.pow(class_imb_ratio, 1 / (self.num_classes - 1))
            for c in range(self.num_classes):
                num_picked = int(
                    len(pseudo_label_list[c]) *
                    np.math.pow(np.math.pow(mu, (self.num_classes - 1) - c), 1 / self.sampling_alpha))  # this is correct!!!
                idx_picked = pseudo_label_list[c][:num_picked]
                data_picked.append(eval_ulb_data[idx_picked])
                targets_picked.append(np.ones_like(eval_ulb_targets[idx_picked]) * c)
                print('class {} is added {} pseudo labels'.format(c, num_picked))
            data_picked.append(lb_data)
            targets_picked.append(lb_targets)
            lb_data = np.concatenate(lb_data, axis=0)
            lb_targets = np.concatenate(lb_targets, axis=0)
        else:
            self.print_fn('Labeled data not update')
        return lb_data, lb_targets

    def set_dataset(self, pseudo_label_list):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        
        # set include_lb_to_ulb to False
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, include_lb_to_ulb=False)
        # eval_ulb
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
        dataset_dict['train_ulb'].data = np.concatenate([dataset_dict['train_lb'].data, dataset_dict['train_ulb'].data])
        dataset_dict['train_ulb'].targets = np.concatenate([dataset_dict['train_lb'].targets, dataset_dict['train_ulb'].targets])

        # add pseudo labels into lb
        lb_data, lb_targets = dataset_dict['train_lb'].data, dataset_dict['train_lb'].targets
        eval_ulb_data, eval_ulb_targets = dataset_dict['eval_ulb'].data, dataset_dict['eval_ulb'].targets
        lb_data, lb_targets = self.gen_split(lb_data, lb_targets, eval_ulb_data, eval_ulb_targets, pseudo_label_list)
        dataset_dict['train_lb'].data = lb_data
        dataset_dict['train_lb'].targets = lb_targets

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.logger.info("unlabeled data number: {}, labeled data number {}, unlabeled eval data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len, len(dataset_dict['eval_ulb'])))
        
        
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict
    
    def set_data_loader(self):
        loader_dict = super().set_data_loader()
        
        # add unlabeled evaluation data loader
        loader_dict['eval_ulb'] = get_data_loader(self.args,
                                                  self.dataset_dict['eval_ulb'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)

        return loader_dict

    def re_init(self, pseudo_label_list):
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.ema = None
        
        # build dataset
        self.dataset_dict = self.set_dataset(pseudo_label_list)

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

        # pseudo_label_list
        pseudo_label_list = None

        for gen in range(self.start_gen, self.num_gens):
            self.gen = gen
            cur = gen / ( self.num_gens - 1)
            cur_dist_align_t = (1.0 - cur) * 1.0 + cur * self.dist_align_t
            # TODO: use cur_dist_align_t in distribution alignment

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
            for key, item in eval_dict.items():
                self.print_fn(f"Generation {gen}, Model result - {key} : {item}")

            # load best model and evaluate on unlabeled dataset to update pseudo_label_list for updating labeled data in CReST
            # TODO: change best_model_path
            best_model_path = os.path.join(self.args.save_dir, self.args.save_name, 'model_best.pth')
            self.load_model(best_model_path)
            ulb_logits = self.evaluate('eval_ulb')['eval_ulb/logits']
            if isinstance(ulb_logits, np.ndarray):
                ulb_logits = torch.from_numpy(ulb_logits)
            ulb_score, ulb_pred = torch.max(ulb_logits, dim=1)
            pseudo_label_list = []
            for c in range(self.num_classes):
                idx = torch.nonzero(ulb_pred == c).view(-1)
                score = ulb_score[idx]
                _, order = score.sort(descending=True)
                idx = idx[order]
                pseudo_label_list.append(idx.numpy())
        
        # TODO: final evluation
        eval_dict = {'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it}
        if 'test' in self.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(self.args.save_dir, self.args.save_name, 'model_best.pth')
            self.load_model(best_model_path)
            test_dict = self.evaluate('test')
            eval_dict['test/best_acc'] = test_dict['test/top-1-acc']
        return eval_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['gen'] = self.gen
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.gen = checkpoint['gen']
        return checkpoint

    @staticmethod
    def get_argument():
        argument = super().get_argument()
        argument.extend([
            SSL_Argument('--num_gens', int, 6),
            SSL_Argument('--dsit_align_t', float, 0.5),
            SSL_Argument('--sampling_alpha', float, 3),
        ])
        return argument

        

# TODO: add register here
def create_crest(CoreAlgorithm, *args, **kwargs):
    class DummyClass(CReST, CoreAlgorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    return DummyClass(*args, **kwargs)

# if __name__ == '__main__':
#     from semilearn.algorithms.fixmatch import FixMatch
#     alg = crest(FixMatch)
#     print(alg)