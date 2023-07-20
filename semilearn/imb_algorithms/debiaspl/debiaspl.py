# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .utils import DebiasPLConsistencyLoss, DebiasPLHook

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@IMB_ALGORITHMS.register('debiaspl')
class DebiasPL(ImbAlgorithmBase):
    """
        DebiasPL algorithm (https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Debiased_Learning_From_Naturally_Imbalanced_Pseudo-Labels_CVPR_2022_paper.pdf).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - debiaspl_tau (float):
                tau in DebiasPl
            - debiaspl_ema_p (float):
                momentum 
    """
    def __init__(self, args, **kwargs):
        self.imb_init(args.debiaspl_tau, args.debiaspl_ema_p)
        super().__init__(args, **kwargs)
        assert args.algorithm not in ['mixmatch', 'meanteacher', 'pimodel'], "DebiasPL not supports {} as the base algorithm.".format(args.algorithm)

        self.p_hat = torch.ones((self.num_classes, )).to(self.gpu) / self.num_classes
        self.consistency_loss = DebiasPLConsistencyLoss(tau=self.tau)


    def imb_init(self, tau=0.4, ema_p=0.999):
        self.tau = tau 
        self.ema_p = ema_p

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(DebiasPLHook(), "NORMAL")


    def compute_prob(self, logits):
        # update p_hat
        probs = super().compute_prob(logits)
        delta_p = probs.mean(dim=0)
        self.p_hat = self.ema_m * self.p_hat + (1 - self.ema_p) * delta_p
        return super().compute_prob(logits - self.tau * torch.log(self.p_hat))

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--debiaspl_tau', float, 0.4),
            SSL_Argument('--debiaspl_ema_p', float, 0.999),
        ]
