'''
wrapper for ReLU
'''

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class ReLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.relu(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
