import torch
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

class Rehearsal():
    def __init__(self, patterns_per_class, patterns_per_class_per_batch=0):
        """
        :param patterns_per_class_per_batch:
            if <0 -> concatenate patterns to the entire dataloader
            if 0 -> concatenate to the current batch another batch size
                    split among existing classes
            if >0 -> concatenate to the current batch `patterns_per_class_per_batch`
                    patterns for each existing class
        """

        self.patterns_per_class = patterns_per_class
        self.patterns_per_class_per_batch = patterns_per_class_per_batch
        self.add_every_batch = patterns_per_class_per_batch >= 0

        self.patterns = {}

    def record_patterns(self, dataloader):
        """
        Update rehearsed patterns with the current data
        """

        counter = defaultdict(int)
        for x,y in dataloader:
            # loop over each minibatch
            for el, _t in zip(x,y):
                t = _t.item()
                if t not in self.patterns:
                    self.patterns[t] = el.unsqueeze(0).clone()
                    counter[t] += 1
                elif counter[t] < self.patterns_per_class:
                    self.patterns[t] = torch.cat( (self.patterns[t], el.unsqueeze(0).clone()) )
                    counter[t] += 1
    

    def concat_to_batch(self, x,y):
        """
        Concatenate subset of memory to the current batch.
        """
        if not self.add_every_batch or self.patterns == {}:
            return x, y

        # how many replay patterns per class per batch?
        # either patterns_per_class_per_batch
        # or batch_size split equally among existing classes
        to_add = int(y.size(0) / len(self.patterns.keys())) \
                if self.patterns_per_class_per_batch == 0 \
                else self.patterns_per_class_per_batch

        rehe_x, rehe_y = [x], [y]
        for k,v in self.patterns.items():
            if to_add >= v.size(0):
                # take directly the memory
                rehe_x.append(v)
            else:
                # select at random from memory
                subset = v[torch.randperm(v.size(0))][:to_add]
                rehe_x.append(subset)
            rehe_y.append(torch.ones(rehe_x[-1].size(0)).long() * k)

        return torch.cat(rehe_x, dim=0), torch.cat(rehe_y, dim=0)


    def _tensorize(self):
        """
        Put the rehearsed pattern into a TensorDataset
        """

        x = []
        y = []
        for k, v in self.patterns.items():
            x.append(v)
            y.append(torch.ones(v.size(0)).long() * k)

        x, y = torch.cat(x), torch.cat(y)

        return TensorDataset(x, y)

    def augment_dataset(self, dataloader):
        """
        Add rehearsed pattern to current dataloader
        """
        if self.add_every_batch or self.patterns == {}:
            return dataloader
        else:
            return DataLoader( ConcatDataset((
                    dataloader.dataset, 
                    self._tensorize()
                )), shuffle=True, drop_last=True, batch_size=dataloader.batch_size)