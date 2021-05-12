import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    '''
    Manipulate images to make them work with RNNs (batch-first) and CNNs.
    '''

    def __init__(self, x,y, permutation=None, pixel_in_input=1, normalization=None, task_vector=None, return_sequences=True):
        '''
        :param x: tensor data
        :param y: tensor label
        :param permutation: apply a permutation on the input, if not None.
                            i.e. `torch.randperm(n_pixels_per_channel)`.
        :param pixel_in_input: reshape input in batch-first (1, -1, pixel_in_input). Do not reshape if `pixel_in_input` is None.
                            e.g. `pixel_in_input=#channels` produces pixel-by-pixel images
        :param normalization: if not None, normalize input data by dividing for `normalization`.
        '''

        self.x = x
        self.y = y
        self.permutation = permutation
        self.pixel_in_input = pixel_in_input
        self.normalization = normalization
        self.task_vector = task_vector
        self.return_sequences = return_sequences

    def __getitem__(self, idxs):

        # (H, W) for gray-scaled images
        x_cur = self.x[idxs].float()
        y_cur = self.y[idxs]

        if self.normalization:
            x_cur = x_cur / float(self.normalization)

        x_cur = x_cur.view(1, -1, self.pixel_in_input)

        if self.permutation is not None:
            x_cur = torch.gather(x_cur, 1,
                self.permutation.unsqueeze(0).repeat(x_cur.size(0),1).unsqueeze(2).repeat(1,1,x_cur.size(2)) ) 

        x_cur = x_cur.squeeze(0) # (len_sequence, pixel_in_input)
        if not self.return_sequences:
            x_cur = x_cur.view(-1)

        if self.task_vector is not None:
            if self.return_sequences:
                x_cur = torch.cat((self.task_vector.unsqueeze(0).repeat(x_cur.size(0),1), x_cur), dim=1)
            else:
                x_cur = torch.cat((self.task_vector, x_cur), dim=0)

        return x_cur, y_cur

    def __len__(self):
        return self.x.size(0)
