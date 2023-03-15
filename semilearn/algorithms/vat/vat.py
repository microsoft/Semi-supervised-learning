# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('vat')
class VAT(AlgorithmBase):
    """
        Virtual Adversarial Training algorithm (https://arxiv.org/abs/1704.03976).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
        - vat_eps ('float',  *optional*, defaults to 6):
            Perturbation size for VAT
        - vat_embd ('bool', *optional*, defaults to False):
            Vat perturbation on word embeddings
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # remixmatch specified arguments
        self.init(unsup_warm_up=args.unsup_warm_up, vat_eps=args.vat_eps, vat_embed=args.vat_embed)
        self.lambda_ent = args.ent_loss_ratio

    def init(self, unsup_warm_up=0.4, vat_eps=6, vat_embed=False):
        self.unsup_warm_up = unsup_warm_up
        self.vat_eps = vat_eps
        self.vat_embed = vat_embed

    def train_step(self, x_lb, y_lb, x_ulb_w):

        with self.amp_cm():
            logits_x_lb = self.model(x_lb)['logits']
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            if self.vat_embed:
                self.bn_controller.freeze_bn(self.model)
                outs_x_ulb_w = self.model(x_ulb_w, return_embed=True)
                ul_x_embed, ul_y  = outs_x_ulb_w['embed'], outs_x_ulb_w['logits']
                # ul_x_embed, ul_y = self.model(x_ulb_w, return_embed=True)
                unsup_loss = self.vat_loss(self.model, x_ulb_w, ul_y, eps=self.vat_eps, ul_x_embed=ul_x_embed, vat_embed=True)
                self.bn_controller.unfreeze_bn(self.model)
            else:
                self.bn_controller.freeze_bn(self.model)
                ul_y = self.model(x_ulb_w)['logits']
                unsup_loss = self.vat_loss(self.model, x_ulb_w, ul_y, eps=self.vat_eps)
                self.bn_controller.unfreeze_bn(self.model)

            loss_entmin = self.entropy_loss(ul_y)
        
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup + self.lambda_ent * loss_entmin

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         loss_entmin=loss_entmin.item())
        return out_dict, log_dict

    def vat_loss(self, model, ul_x, ul_y, xi=1e-6, eps=6, num_iters=1, ul_x_embed=None, vat_embed=False):
        # find r_adv
        if vat_embed:
            d = torch.Tensor(ul_x_embed.size()).normal_()
        else:
            d = torch.Tensor(ul_x.size()).normal_()
            
        for i in range(num_iters):
            d = xi * self._l2_normalize(d)
            d = Variable(d.cuda(self.gpu), requires_grad=True)

            if vat_embed:
                y_hat = model({'attention_mask': ul_x['attention_mask'],
                               'inputs_embeds': ul_x_embed.detach() + d}, return_embed=True)['logits']
            else:
                y_hat = model(ul_x + d)['logits']

            delta_kl = self.kl_div_with_logit(ul_y.detach(), y_hat)
            delta_kl.backward()

            d = d.grad.data.clone().cpu()
            model.zero_grad()

        d = self._l2_normalize(d)
        d = Variable(d.cuda(self.gpu))
        r_adv = eps * d
        # compute lds

        if vat_embed:
            y_hat = model({'attention_mask': ul_x['attention_mask'],
                           'inputs_embeds': ul_x_embed + r_adv.detach()}, return_embed=True)['logits']
        else:
            y_hat = model(ul_x + r_adv.detach())['logits']

        delta_kl = self.kl_div_with_logit(ul_y.detach(), y_hat)
        return delta_kl

    def _l2_normalize(self, d):
        # TODO: put this to cuda with torch
        d = d.numpy()
        if len(d.shape) == 4:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        elif len(d.shape) == 3:
            d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
        return torch.from_numpy(d)

    def kl_div_with_logit(self, q_logit, p_logit):

        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1).mean(dim=0)
        qlogp = (q * logp).sum(dim=1).mean(dim=0)

        return qlogq - qlogp

    def entropy_loss(self, ul_y):
        p = F.softmax(ul_y, dim=1)
        return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--ent_loss_ratio', float, 0.06, 'Entropy minimization weight'),
            SSL_Argument('--vat_eps', float, 6, 'VAT perturbation size.'),
            SSL_Argument('--vat_embed', str2bool, False, 'use word embedding for vat, specified for nlp'),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
        ]
