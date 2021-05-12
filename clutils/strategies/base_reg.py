import torch
from collections import defaultdict
from .utils import padded_op, copy_params_dict


class BaseReg():
    def __init__(self, model, device, lamb=1, 
            normalize=True, single_batch=False, cumulative='none'):
        '''

        Task IDs are ordered integer (0,1,2,3,...)

        :param lamb: task specific hyper-parameter for the penalization.
        :param normalize: normalize final importance matrix in [0,1] (normalization computed block wise).
        :param single_batch: if True compute fisher by averaging gradients pattern by pattern. 
                If False, compute fisher by averaging mini batches.
        :param cumulative: Keep a single penalty matrix for all tasks or one for each task.
                Possible values are 'none' or 'sum', 'avgclip'.
                'avgclip' = clip((F_{i-1} + F_i) / i, max_f)
                Keep one separate penalty for each task if 'none'. 

        '''

        self.model = model
        self.lamb = lamb
        self.device = device
        self.normalize = normalize
        self.single_batch = single_batch
        assert(cumulative == 'none' or cumulative == 'sum' or cumulative == 'avgclip')
        self.cumulative = cumulative

        self.saved_params = defaultdict(list)
        self.importance = defaultdict(list)

        self.max_clip = 0.8
        self.wi = {0: 0.00015, 1: 0.000005, 2: 0.000005, 3: 0.000005, 4: 0.000005}

    def penalty(self, current_task_id):
        '''
        Compute regularization penalty.
        Sum the contribution over all tasks if importance is not cumulative.

        :param current_task_id: current task ID (0 being the first task)
        '''

        total_penalty = torch.zeros(1, dtype=torch.float32, device=self.device).squeeze()

        if self.cumulative == 'none':
            for task in range(current_task_id):
                for (_, param), (_, saved_param), (_, imp) in zip(self.model.named_parameters(), self.saved_params[task], self.importance[task]):
                    saved_param = saved_param.to(self.device)
                    imp = imp.to(self.device)
                    pad_difference = padded_op(param, saved_param)
                    total_penalty += (padded_op(imp, pad_difference.pow(2), op='*')).sum()
        elif (self.cumulative == 'sum' or self.cumulative == 'avgclip') and current_task_id > 0:
            for (_, param), (_, saved_param), (_, imp) in zip(self.model.named_parameters(), self.saved_params[current_task_id], self.importance[current_task_id]):
                saved_param = saved_param.to(self.device)
                imp = imp.to(self.device)
                pad_difference = padded_op(param, saved_param)
                total_penalty += (padded_op(imp, pad_difference.pow(2), op='*')).sum()            

        return self.lamb * total_penalty
        

    def save_old_parameters(self, current_task_id):
        # store learned parameters and importance coefficients
        # no need to store all the tensor metadata, just its data (data.clone())
        self.saved_params[current_task_id] = copy_params_dict(self.model, to_cpu=True)

    def update_importance(self, current_task_id, importance, save_pars=True):
        '''
        :param current_task_id: current task ID (0 being the first task)
        :importance : importance for each weight
        '''
        
        if save_pars:
            self.save_old_parameters(current_task_id)
 
        if self.cumulative == 'none' or current_task_id == 0:
            self.importance[current_task_id] = importance

        elif self.cumulative == 'sum' and current_task_id > 0:
            self.importance[current_task_id] = []
            for (k1,curr_imp),(k2,imp) in zip(self.importance[current_task_id-1], importance):
                assert(k1==k2)
                self.importance[current_task_id].append( (k1, padded_op(imp, curr_imp, op='+')) )
        elif self.cumulative == 'avgclip' and current_task_id > 0:
            for (k1,curr_imp),(k2,imp) in zip(self.importance[current_task_id-1], importance):
                assert(k1==k2)
                self.importance[current_task_id].append( (
                    # self.wi[current_task_id]*curr_imp
                    k1, (padded_op(imp, curr_imp, op='+') / float(current_task_id+1)).clamp(max=self.max_clip)
                ))