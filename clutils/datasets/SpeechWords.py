import torch
import os
from torchaudio import transforms
from torch.utils.data import TensorDataset, DataLoader
from .utils import get_fixed_train_val_test

class CLSpeechWords():
    def __init__(self, root, train_batch_size, test_batch_size, n_mels=40, perc_test=0.2,
                 len_task_vector=0, task_vector_at_test=False, return_sequences=True):

        self.root = root
        self.perc_test = perc_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.return_sequences = return_sequences
        self.len_task_vector = len_task_vector
        self.task_vector_at_test = task_vector_at_test

        self.sample_rate = 16000

        win_length = int(self.sample_rate / 1000 * 25)
        hop_length = int(self.sample_rate / 1000 * 10)
        self.mel_spectr = transforms.MelSpectrogram(sample_rate=self.sample_rate, 
            win_length=win_length, hop_length=hop_length, n_mels=n_mels)

        self.dataloaders = []

        self.current_class_id = 0
            

    def _load_data(self, classes):
        features = []
        targets = []
        for classname in classes:
            feature = torch.load(os.path.join(self.root, 'pickled', f"{classname}.pt"))
            features.append( self.preprocess_wav(feature) )
            # normalization
            #mean = train_x.view(-1, train_x.size(-1)).mean(dim=0) # mean over all batches -> F
            #std = train_x.view(-1, train_x.size(-1)).std(dim=0)
            #train_x = (train_x - mean) / std
            targets.append(torch.ones(features[-1].size(0)).long() * self.current_class_id)
            self.current_class_id += 1
            
        features = torch.cat(features) # (B, L, n_mel)
        targets = torch.cat(targets)

        return features, targets


    def preprocess_wav(self, wav):
        features = self.mel_spectr(wav).permute(0, 2, 1) # B, L, F
        return features


    def get_task_loaders(self, classes=None, task_id=None):

        if classes is not None:
            X, Y = self._load_data(classes)

            if not self.return_sequences:
                X = X.view(X.size(0), -1)

            data_len = Y.size(0)
            len_train = int(data_len - data_len  * self.perc_test)
            len_test = data_len - len_train

            indices = torch.randperm(data_len)
            train_x, test_x = X[indices[:len_train]], X[indices[len_train:]]
            train_y, test_y = Y[indices[:len_train]], Y[indices[len_train:]]

            if self.len_task_vector > 0:
                task_vector_zeros = torch.zeros(self.len_task_vector).float()
                task_vector_train = task_vector_zeros.clone()
                task_vector_train[len(self.dataloaders)] = 1.

                if self.return_sequences:
                    train_x = torch.cat((task_vector_train.unsqueeze(0).unsqueeze(0).repeat(train_x.size(0),train_x.size(1), 1),
                                        train_x), dim=2)
                    if self.task_vector_at_test:
                        test_x = torch.cat((task_vector_train.unsqueeze(0).unsqueeze(0).repeat(test_x.size(0),test_x.size(1), 1),
                                            test_x), dim=2)
                    else:
                        test_x = torch.cat((task_vector_zeros.unsqueeze(0).unsqueeze(0).repeat(test_x.size(0),test_x.size(1), 1),
                                            test_x), dim=2)
                else:
                    train_x = torch.cat((task_vector_train.unsqueeze(0).repeat(train_x.size(0),1), train_x), dim=1)
                    if self.task_vector_at_test:
                        test_x = torch.cat((task_vector_train.unsqueeze(0).repeat(test_x.size(0), 1), test_x), dim=1)
                    else:
                        test_x = torch.cat((task_vector_zeros.unsqueeze(0).repeat(test_x.size(0), 1), test_x), dim=1)


            train_d = TensorDataset(train_x, train_y)
            test_d = TensorDataset(test_x, test_y)
            train_batch_size = len(train_d) if self.train_batch_size == 0 else self.train_batch_size
            test_batch_size = len(test_d) if self.test_batch_size == 0 else self.test_batch_size

            train_loader = DataLoader(train_d, batch_size=train_batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_d, batch_size=test_batch_size, shuffle=False, drop_last=True)
            self.dataloaders.append( [train_loader, test_loader] )
            return train_loader, test_loader

        elif task_id is not None:
            return self.dataloaders[task_id]


class SpeechWords():
    def __init__(self, root, train_batch_size, test_batch_size, n_mels=40, perc_test=0.2,
                 return_sequences=True):

        self.root = root
        self.perc_test = perc_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.return_sequences = return_sequences

        self.sample_rate = 16000

        win_length = int(self.sample_rate / 1000 * 25)
        hop_length = int(self.sample_rate / 1000 * 10)
        self.mel_spectr = transforms.MelSpectrogram(sample_rate=self.sample_rate,
            win_length=win_length, hop_length=hop_length, n_mels=n_mels)

        self.current_class_id = 0

    def _load_data(self, classes):
        train_features, val_features, test_features = [], [], []
        train_targets, val_targets, test_targets = [], [], []

        for classname in classes:
            feature = torch.load(os.path.join(self.root, 'pickled', f"{classname}.pt"))
            feature = self.preprocess_wav(feature)
            train_feature, val_feature, test_feature = get_fixed_train_val_test(feature, self.perc_test)
            train_features.append(train_feature)
            val_features.append(val_feature)
            test_features.append(test_feature)
            train_targets.append(torch.ones(train_features[-1].size(0)).long() * self.current_class_id)
            val_targets.append(torch.ones(val_features[-1].size(0)).long() * self.current_class_id)
            test_targets.append(torch.ones(test_features[-1].size(0)).long() * self.current_class_id)

            self.current_class_id += 1

        train_features = torch.cat(train_features) # (B, L, n_mel)
        val_features = torch.cat(val_features) # (B, L, n_mel)
        test_features = torch.cat(test_features) # (B, L, n_mel)
        train_targets = torch.cat(train_targets)
        val_targets = torch.cat(val_targets)
        test_targets = torch.cat(test_targets)

        return train_features, train_targets, val_features, val_targets, test_features, test_targets


    def preprocess_wav(self, wav):
        features = self.mel_spectr(wav).permute(0, 2, 1) # B, L, F
        return features


    def get_loaders(self, classes):

        train_x, train_y, val_x, val_y, test_x, test_y = self._load_data(classes)

        if not self.return_sequences:
            train_x = train_x.view(train_x.size(0), -1)
            val_x = val_x.view(val_x.size(0), -1)
            test_x = test_x.view(test_x.size(0), -1)

        train_d = TensorDataset(train_x, train_y)
        val_d = TensorDataset(val_x, val_y)
        test_d = TensorDataset(test_x, test_y)
        train_batch_size = len(train_d) if self.train_batch_size == 0 else self.train_batch_size
        test_batch_size = len(test_d) if self.test_batch_size == 0 else self.test_batch_size

        train_loader = DataLoader(train_d, batch_size=train_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_d, batch_size=test_batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_d, batch_size=test_batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader, test_loader
