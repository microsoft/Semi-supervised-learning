'''
Implementation of Brier Score.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BrierScore(nn.Module):
    def __init__(self):
        super(BrierScore, self).__init__()

    def forward(self, input, target,idx=None):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = F.softmax(input)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(input.shape[0])
        return loss