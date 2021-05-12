import torch
from copy import deepcopy
from ..base_reg import BaseReg
from ..utils import normalize_blocks, zerolike_params_dict


class EWC(BaseReg):
    def __init__(self, model, device, lamb=1, 
            normalize=True, single_batch=False, cumulative='none'):
        '''

        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for EWC.
        :param normalize: normalize final fisher matrix in [0,1] (normalization computed among all parameters).
        :param single_batch: if True compute fisher by averaging gradients pattern by pattern. 
                If False, compute fisher by averaging mini batches.
        :param cumulative: possible values are 'none', 'sum'.
                Keep one separate penalty for each task if 'none'. 

        '''

        super(EWC, self).__init__(model, device, lamb, normalize, single_batch, cumulative)


    def compute_importance(self, optimizer, criterion, task_id, loader,
            update=True, truncated_time=0, compute_for_head=True):
        '''
        :param update: update EWC structure with final fisher
        :truncated_time: 0 to compute gradients along all the sequence
                A positive value to use only last `truncated_time` sequence steps.
        :compute_for_head: compute importance also for output layer
        '''

        self.model.train()

        # list of list
        fisher_diag = zerolike_params_dict(self.model, to_cpu=True)

        for i, (x,y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)

            if self.single_batch:
                for b in range(x.size(0)):
                    x_cur = x[b].unsqueeze(0)
                    y_cur = y[b].unsqueeze(0)

                    optimizer.zero_grad()
                    if truncated_time > 0:
                        out = self.model(x_cur, truncated_time=truncated_time)
                    else:
                        out = self.model(x_cur)
                    loss = criterion(out, y_cur)
                    loss.backward()
                    for (k1,p),(k2,f) in zip(self.model.named_parameters(), fisher_diag):
                        assert(k1==k2)
                        if compute_for_head or (not k1.startswith('layers.out')):
                                f += p.grad.cpu().data.clone().pow(2)
            else:
                optimizer.zero_grad()
                if truncated_time > 0:
                    out = self.model(x, truncated_time=truncated_time)
                else:
                    out = self.model(x)
                loss = criterion(out, y)
                loss.backward()

                for (k1,p),(k2,f) in zip(self.model.named_parameters(), fisher_diag):
                    assert(k1==k2)
                    if compute_for_head or (not k1.startswith('layers.out')):
                        f += p.grad.cpu().data.clone().pow(2)
        
        for _, f in fisher_diag:
            
            if self.single_batch:
                f /= ( float(x.size(0)) * float(len(loader)))
            else:
                f /= float(len(loader))

        unnormalized_fisher = deepcopy(fisher_diag)

        # max-min normalization among every parameter group
        if self.normalize:
            fisher_diag = normalize_blocks(fisher_diag)

        if update:
            self.update_importance(task_id, fisher_diag)


        return fisher_diag, unnormalized_fisher
