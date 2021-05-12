from torch.utils.data import random_split, ConcatDataset
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def get_fixed_train_val_test(features, perc_test):
    num_el = features.size(0)
    train_size = num_el - int(num_el * (perc_test*2))
    test_size = int(num_el*perc_test)

    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size+test_size))
    test_idx = list(range(train_size+test_size, train_size+(2*test_size)))

    train_f, val_f, test_f = features[train_idx], features[val_idx], features[test_idx]

    return train_f, val_f, test_f

def split_dataset(dataset, l1, l2):
    split_list = [int(l1), int(l2)]
    split_datasets = random_split(dataset, split_list)
    return split_datasets

def merge_datasets(dataset_list):
    """
    List of PyTorch Dataset
    """

    return ConcatDataset(dataset_list)

def compute_quickdraw_normalizer(root):
    classes = [ el for el in os.listdir(root) \
                if not os.path.isdir(el) and not el.endswith('.full.npz')
                and not el.endswith('.png')]

    normalizers = {}

    for cls in classes:
        data = np.load(os.path.join(root, cls), encoding='latin1', allow_pickle=True)
        data = data['train']
        deltas = []
        for i in range(len(data)):
            deltas += data[i][:, 0].tolist()
            deltas += data[i][:, 1].tolist()
        std = np.std(deltas)
        mu = np.mean(deltas)
        normalizers[os.path.splitext(os.path.basename(cls))[0]] = (mu, std)

    return normalizers

def collate_sequences(x):
    x_features = [el[0] for el in x]
    y = torch.stack([el[1] for el in x], dim=0).long()
    lengths = [el.size(0) for el in x_features]
    x_padded = pad_sequence(x_features, batch_first=True)
    return x_padded, y
