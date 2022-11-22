# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy

from semilearn.core import ImbAlgorithmBase
from semilearn.algorithms.utils import SSL_Argument
from semilearn.core.utils import get_data_loader, IMB_ALGORITHMS
from .utils import AdaptiveThresholdingHook


@IMB_ALGORITHMS.register('adsh')
class Adsh(ImbAlgorithmBase):
    """
        Adsh algorithm (https://proceedings.mlr.press/v162/guo22e/guo22e.pdf).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - adsh_tau_1 (`float`):
                threshold for adsh
    """
    def __init__(self, args, **kwargs):
        self.imb_init(tau_1=args.adsh_tau_1)
        super().__init__(args, **kwargs)
        assert args.algorithm == 'fixmatch', "Adsh only supports FixMatch as the base algorithm."

    def imb_init(self, tau_1):
        self.tau_1 = tau_1

    def set_dataset(self):
        dataset_dict = super().set_dataset()
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
        dataset_dict['eval_ulb'].is_ulb = False
        return dataset_dict

    def set_data_loader(self):
        loader_dict = super().set_data_loader()

        # add unlabeled evaluation data loader
        loader_dict['eval_ulb'] = get_data_loader(self.args,
                                                  self.dataset_dict['eval_ulb'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)

        return loader_dict

    def set_hooks(self):
        super().set_hooks()

        # reset hooks
        self.register_hook(AdaptiveThresholdingHook(self.num_classes, self.tau_1), "MaskingHook")

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--adsh_tau_1', float, 0.95),
        ]
