import torch
from collections import defaultdict
from copy import deepcopy
from ..base_reg import BaseReg
from ..utils import normalize_blocks, zerolike_params_dict


class MAS(BaseReg):
    """
    Memory Aware Synapses
    """

    def __init__(self, model, device, lamb=1):
        '''
        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for Memory Aware Synapses.
        :param normalize: normalize final importance matrix in [0,1] (normalization computed among all parameters).
        '''

        super(MAS, self).__init__(model, device, lamb, cumulative='sum')

        self.reset_current_importance()

    def reset_current_importance(self):
        self.current_importance = zerolike_params_dict(self.model, to_cpu=True)
        self.patterns_so_far = 0

    def compute_importance(self, optimizer, x, truncated_time=0):
        '''
        :param update: Update MAS importance
        '''

        self.model.train()

        optimizer.zero_grad()
        if truncated_time > 0:
            out = self.model(x, truncated_time=truncated_time)
        else:
            out = self.model(x)
        loss = out.norm(p=2).pow(2)
        loss.backward()

        for (k1,p),(k2,imp) in zip(self.model.named_parameters(), self.current_importance):
            assert(k1==k2)
            imp *= self.patterns_so_far
            imp += p.grad.cpu().data.clone().abs()
            self.patterns_so_far += x.size(0)
            imp /= float(self.patterns_so_far)
