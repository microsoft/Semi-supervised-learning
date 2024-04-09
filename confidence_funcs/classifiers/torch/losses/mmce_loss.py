'''
Implementation of the MMCE (MMCE_m) and MMCE_weighted (MMCE_w).
Reference:
[1]  A. Kumar, S. Sarawagi, U. Jain, Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings.
     ICML, 2018.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MMCE(nn.Module):
    """
    Computes MMCE_m loss.
    """
    def __init__(self, loss_conf = {'device':"cuda:0"}):
        super(MMCE, self).__init__()
        self.device = loss_conf['device']
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mmce_coeff = loss_conf['mmce_coeff']
        self.epsilon = 1e-12
        self.result = {}

    def kernel(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        diff = c1[:, None] - c2
        return torch.exp(-2.5 * torch.abs(diff))

    def forward(self, input, target, idx=None):
        probs, labels = torch.max(torch.softmax(input, dim=1), dim=1)
        probs = torch.clamp(probs, min=self.epsilon, max=1. - self.epsilon)

        matched = torch.where(labels == target, torch.ones_like(labels), torch.zeros_like(labels))
        n_samples = len(matched)
        n_correct = torch.sum(matched)

        # divide all probabilities by matched/not matched
        probs_false = probs[matched == 0]
        probs_correct = probs[matched == 1]

        # compute kernels between different combinations
        kernel_false = self.kernel(probs_false, probs_false)
        kernel_correct = self.kernel(probs_correct, probs_correct)
        kernel_mixed = self.kernel(probs_correct, probs_false)

        probs_false = torch.unsqueeze(probs_false, dim=1)
        inv_probs_correct = torch.unsqueeze(1. - probs_correct, dim=1)

        diff_false = torch.matmul(probs_false, probs_false.transpose(1, 0))
        diff_correct = torch.matmul(inv_probs_correct, inv_probs_correct.transpose(1, 0))
        diff_mixed = torch.matmul(inv_probs_correct, probs_false.transpose(1, 0))

        # MMCE calculation scheme (see paper for mathematical details)
        part_false = torch.sum(diff_false * kernel_false) / float((n_samples - n_correct) ** 2) if n_samples - n_correct > 0 else 0.
        part_correct = torch.sum(diff_correct * kernel_correct) / float(n_correct ** 2) if n_correct > 0 else 0.
        part_mixed = 2 * torch.sum(diff_mixed * kernel_mixed) / float((n_samples - n_correct) * n_correct) if (n_samples - n_correct) * n_correct > 0 else 0.

        mmce = self.mmce_coeff * torch.sqrt(part_false + part_correct - part_mixed)
        loss = mmce + self.cross_entropy_loss(input, target)
        self.result['loss'] = loss.item()

        return loss
    
    def batch_closure_callback(self, batch_state):
        pass
    
    def epoch_closure_callback(self, epoch_state):
        pass