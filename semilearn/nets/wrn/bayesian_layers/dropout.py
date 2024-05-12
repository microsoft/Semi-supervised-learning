'''
wrapper for Dropout 
'''

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class Dropout(nn.Module):
    __constants__ = ['p', 'inplace']

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".
                format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.dropout(input[0], self.p, self.training, self.inplace), kl

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.p, self.inplace)
