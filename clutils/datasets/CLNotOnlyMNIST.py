import torch
import os
import pickle
import numpy as np
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
from functools import partial
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .ImageDataset import ImageDataset
from .utils import split_dataset


class _CLImageDataset(Dataset):
    '''
    General class providing useful methods to manage subsets of classes and permutations.
    Must be inherited by a subclass as second parent class together with the target PyTorch dataset class.
    '''

    def __init__(self, pixel_in_input=1, perc_test=0.2, train_batch_size=3, test_batch_size=0,
            max_label_value=None, sequential=False, image_size=(28,28), normalization=None,
            len_task_vector=0, task_vector_at_test=False, return_sequences=True):
        '''
        :param perc_test: percentage of dataset used for test
        :param train_batch_size, test_batch_size: 0 to use a full batch, otherwise > 0.
        :param sequential: True to use pixel-wise images. False to use permuted pixel-wise images.
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        :param len_task_vector: concatenate to the input a one-hot task vector if len_task_vector > 0.
        :param task_vector_at_test: concatenate task vector also at test time.
        :param return_sequences: if True returns (B,L,I) else (B,I)
        '''

        # to be instantiated from children classes
        self.train_data_all, self.train_targets_all = [], []
        self.mytest_data_all, self.mytest_targets_all = [], []

        self.pixel_in_input = pixel_in_input
        self.H, self.W = image_size[0], image_size[1]
        assert( (self.H*self.W) % self.pixel_in_input == 0)

        self.sequential = sequential
        self.normalization = normalization
        self.permutations = []
        self.filter_classes = []

        self.return_sequences = return_sequences

        self.len_task_vector = len_task_vector
        self.task_vector_at_test = task_vector_at_test

        self.perc_test = perc_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_label_value = max_label_value

        self.dataloaders = []

        self._change_permutation()
        
    def _change_permutation(self):

        if not self.sequential:
            perm = torch.randperm( int( (self.H*self.W) / self.pixel_in_input ) )
            self.permutations.append(perm)

    def save_permutation(self, filepath):
        torch.save(self.permutations[-1], os.path.join(filepath, f"permutation{len(self.permutations) - 1}.pt"))

    def _select_digits_subset(self, classes):
        if classes is not None:
            mask_train = torch.nonzero( torch.sum( torch.stack([ (self.train_targets_all == l) for l in classes ]), dim=0), as_tuple=False).squeeze()
            mask_test = torch.nonzero(torch.sum( torch.stack([ (self.mytest_targets_all == l) for l in classes ]), dim=0), as_tuple=False).squeeze()

            self.train_targets = self.train_targets_all[mask_train]
            self.mytrain_data = self.train_data_all[mask_train]
            self.mytest_targets = self.mytest_targets_all[mask_test]
            self.mytest_data = self.mytest_data_all[mask_test]
        else:
            self.train_targets = self.train_targets_all
            self.mytrain_data = self.train_data_all
            self.mytest_targets = self.mytest_targets_all
            self.mytest_data = self.mytest_data_all


    def _get_test_loader(self, perm=None):
        '''
        :return mnist_cl_loader_test: mini or full batch loader for test set, depending on batch_size_test
        '''

        if self.len_task_vector > 0:
            task_vector = torch.zeros(self.len_task_vector).float()
            if self.task_vector_at_test:
                task_vector[len(self.dataloaders)] = 1.
        else:
            task_vector = None

        test_dataset = ImageDataset(self.mytest_data, self.mytest_targets, perm, self.pixel_in_input,
                                    self.normalization, task_vector=task_vector, return_sequences=self.return_sequences)
        if self.test_batch_size == 0:
            return DataLoader(test_dataset, batch_size=len(self.mytest_targets), shuffle=False)
        else:
            return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False, drop_last=True)


    def _get_train_loader(self, perm=None):
        '''
        :return mnist_cl_loader_train: mini-batch loader for training set
        '''

        if self.len_task_vector > 0:
            task_vector = torch.zeros(self.len_task_vector).float()
            task_vector[len(self.dataloaders)] = 1.
        else:
            task_vector = None

        train_dataset = ImageDataset(self.mytrain_data, self.train_targets, perm, self.pixel_in_input,
                                        self.normalization, task_vector=task_vector, return_sequences=self.return_sequences)

        train_batch_size = int(len(self.mytrain_data)) if self.train_batch_size == 0 else self.train_batch_size
        cl_loader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
        return cl_loader_train


    def get_task_loaders(self, classes=None, task_id=None, change_permutation=True):
        '''
        Select a subset of dataset with provided classes and eventually permute images.
        Returns dataloaders for training and test set.

        :param classes: a list containing integer representing digits to select from dataset. If None select all images.
        :param task_id: get dataloaders from a previous task id
        '''

        if task_id is not None:
            return self.dataloaders[task_id]

        # use new permutation for next task
        if not self.sequential:
            if change_permutation:
                self._change_permutation()
            perm = self.permutations[-1]
        else:
            perm = None

        self.filter_classes.append(classes)

        # select classes subset based on digit classes
        self._select_digits_subset(classes)
            
        # restrict output targets to max_label_value values
        if self.max_label_value is not None:
            self.train_targets = self.train_targets % self.max_label_value
            self.mytest_targets = self.mytest_targets % self.max_label_value

        train_loader = self._get_train_loader(perm)
        test_loader = self._get_test_loader(perm)
        self.dataloaders.append( (train_loader, test_loader) )
        return train_loader, test_loader


class CLMNIST(MNIST, _CLImageDataset):
    def __init__(self, root, download, pixel_in_input=1, perc_test=0.2, train_batch_size=3, test_batch_size=0,
            max_label_value=None, sequential=False, image_size=(28,28), normalization=None, len_task_vector=0,
            task_vector_at_test=False, return_sequences=True):
        '''
        :param max_label_value: number of output units of the model
        :param test_batch_size: 0 to use a full batch, otherwise > 0.
        :param sequential: True to use pixel-wise images. False to use permuted pixel-wise images.
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        '''
        
        super(CLMNIST, self).__init__(root, train=True, download=download, transform=transforms.ToTensor())
        _CLImageDataset.__init__(self, pixel_in_input=pixel_in_input, perc_test=perc_test, train_batch_size=train_batch_size, test_batch_size=test_batch_size,
            max_label_value=max_label_value, sequential=sequential, image_size=image_size, normalization=normalization,
            len_task_vector=len_task_vector, task_vector_at_test=task_vector_at_test, return_sequences=return_sequences)

        self.train_data_all, self.train_targets_all = torch.load(os.path.join(self.processed_folder, self.training_file))
        self.mytest_data_all, self.mytest_targets_all = torch.load(os.path.join(self.processed_folder, self.test_file))


def mnist_collate(batch, return_sequences):
    out = []
    y = torch.stack([torch.tensor(el[1]).long() for el in batch], dim=0)
    for x_cur, _ in batch:
        if not return_sequences:
            x_cur = x_cur.view(-1)
        out.append(x_cur)

    out = torch.stack(out, dim=0) if not return_sequences else torch.cat(out, dim=0)

    return out, y

class MMNIST():
    """
    Multitask MNIST both permuted and split
    """
    def __init__(self, root, download, pixel_in_input, perc_val, train_batch_size, test_batch_size,
             return_sequences=True, split=True, sequential=False):

        self.return_sequences = return_sequences
        self.pixel_in_input = pixel_in_input
        mnist_transform = [transforms.ToTensor(),transforms.Lambda(
                    partial(lambda el, pixel_in_input: el.view(1, -1, pixel_in_input),
                            pixel_in_input=self.pixel_in_input))]

        if split:
            N_TASKS = 1
            if not sequential:
                mnist_transform.append(transforms.Lambda(partial(
                    lambda im, p: torch.gather(im, 1,p.unsqueeze(0).unsqueeze(2).repeat(1,1,im.size(2))),
                    p=torch.randperm(int(784/self.pixel_in_input)))))
        else:
            N_TASKS = 10
            permutations = [torch.randperm(int(784/self.pixel_in_input))  for _ in range(N_TASKS) ]

        datasets = {'train': [], 'val': [], 'test': []}

        for i in range(N_TASKS):
            if i > 0:
                del  mnist_transform[-1]
            if not split:
                mnist_transform.append(transforms.Lambda(partial(
                    lambda im, p: torch.gather(im, 1,p.unsqueeze(0).unsqueeze(2).repeat(1,1,im.size(2))),
                    p=permutations[i])))

            train_dataset = MNIST(root, train=True, download=download, transform=transforms.Compose(mnist_transform))
            len_val = int(len(train_dataset) * perc_val)
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - len_val, len_val])
            datasets['train'].append(train_dataset)
            datasets['val'].append(val_dataset)
            datasets['test'].append(
                MNIST(root, train=False, download=download, transform=transforms.Compose(mnist_transform)))

        self.train_dataset = ConcatDataset(datasets['train'])
        self.val_dataset = ConcatDataset(datasets['val'])
        self.test_dataset = ConcatDataset(datasets['test'])

        vbs = test_batch_size if test_batch_size > 0 else len(self.val_dataset)
        tbs = test_batch_size if test_batch_size > 0 else len(self.test_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True,
                                  collate_fn=partial(mnist_collate,
                                                     return_sequences=self.return_sequences))
        self.val_loader = DataLoader(self.val_dataset, batch_size=vbs, shuffle=False, drop_last=True,
                                  collate_fn=partial(mnist_collate,
                                                     return_sequences=self.return_sequences))
        self.test_loader = DataLoader(self.test_dataset, batch_size=tbs, shuffle=False, drop_last=True,
                                  collate_fn=partial(mnist_collate,
                                                     return_sequences=self.return_sequences))
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
