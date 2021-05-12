import torch
import torch.nn as nn
from .utils import expand_output_layer
from ..globals import OUTPUT_TYPE, choose_output
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaRNN(nn.Module):
    """
    Layers are described by the following names:
    'rnn' -> recurrent module
    'out' -> linear readout

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """

    def __init__(self, input_size, hidden_size, output_size, device,
                num_layers=1, dropout=0., bidirectional=False, 
                truncated_time=0, relu=False, orthogonal=False):
        '''
        :param truncated_time: an integer representing the 
            time step to backpropagate (from the end of sequence).
        '''

        super(VanillaRNN, self).__init__()

        self.output_type = OUTPUT_TYPE.LAST_OUT
        self.is_recurrent = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation = 'relu' if relu else 'tanh'
        self.dropout = dropout if self.num_layers > 1 else 0.
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.device = device
        self.orthogonal = orthogonal
        self.truncated_time = truncated_time

        self.layers = nn.ModuleDict([])

        self.layers.update([
            ['rnn', nn.RNN(self.input_size, self.hidden_size,
            num_layers=self.num_layers, nonlinearity=self.activation,
            batch_first=True, dropout=self.dropout,
            bidirectional=self.bidirectional) ]
        ])

        if self.orthogonal:
            for _, hh, _, _ in self.layers['rnn'].all_weights:
                nn.init.orthogonal_(hh)


        self.layers.update([
            ['out', nn.Linear(self.directions*self.hidden_size, self.output_size) ]
        ])

        self.layers = self.layers.to(self.device)


    def forward(self, x, h=None, truncated_time=None):
        '''
        :param x: (batch_size, seq_len, input_size)
        :param h: hidden state of the recurrent module

        :return out: (batch_size, seq_len, directions*hidden_size)
        :return h: hidden state of the recurrent module
        '''

        tr_time = truncated_time if truncated_time else self.truncated_time

        if tr_time > 0:
            with torch.no_grad():
                if h:
                    out_h1, h1 = self.layers['rnn'](x[:, :-tr_time, :], h)
                else:
                    out_h1, h1 = self.layers['rnn'](x[:, :-tr_time, :])

            out_h2, h2 = self.layers['rnn'](x[:, -tr_time:, :], h1)
            out = self.layers['out'](out_h2)
            out_h = torch.cat((out_h1, out_h2), dim=0)

        else:
            if h:
                out_h, h = self.layers['rnn'](x, h)
            else:
                out_h, h = self.layers['rnn'](x)

            out = self.layers['out'](out_h)

        return choose_output(out, out_h, self.output_type)


    def reset_memory_state(self, batch_size):
        '''
        :param batch_size: size of current batch. 
        '''

        h = torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device)

        return h

    def expand_output_layer(self, n_units=2):
        self.layers["out"] = expand_output_layer(self.layers["out"], n_units)


    def get_layers(self):
        return self.layers.values()


class LSTM(nn.Module):
    """
    Layers are described by the following names:
    'rnn' -> recurrent module
    'out' -> linear readout layer

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """

    def __init__(self, input_size, hidden_size, output_size, device,
                num_layers=1, dropout=0., bidirectional=False, 
                truncated_time=0, orthogonal=False):
        '''
        :param truncated_time: an integer representing the 
            time step to backpropagate (from the end of sequence).
        '''

        super(LSTM, self).__init__()

        self.output_type = OUTPUT_TYPE.LAST_OUT
        self.is_recurrent = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0.
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.device = device
        self.orthogonal = orthogonal
        self.truncated_time = truncated_time

        self.layers = nn.ModuleDict([])

        self.layers.update([ 
            ['rnn', nn.LSTM(self.input_size, self.hidden_size, \
                    self.num_layers, batch_first=True, dropout=self.dropout, \
                    bidirectional=self.bidirectional) ]
        ])
        if self.orthogonal:
            for _, hh, _, _ in self.layers['rnn'].all_weights:
                # lstm divides hidden matrix into 4 chunks
                # https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
                for j in range(0, hh.size(0), self.hidden_size): 
                    nn.init.orthogonal_(hh[j:j+self.hidden_size])

        self.layers.update([
            ['out', nn.Linear(self.directions*self.hidden_size, self.output_size) ]
        ])

        self.layers = self.layers.to(self.device)

    def forward(self, x, h=None, truncated_time=None):
        '''
        :param x: (batch_size, seq_len, input_size)
        :param h: hidden state of the recurrent module

        :return out: (batch_size, seq_len, directions*hidden_size)
        :return h: hidden state of the recurrent module
        '''

        tr_time = truncated_time if truncated_time else self.truncated_time

        if tr_time > 0:
            with torch.no_grad():
                if h:
                    out_h1, h1 = self.layers['rnn'](x[:, :-tr_time, :], h)
                else:
                    out_h1, h1 = self.layers['rnn'](x[:, :-tr_time, :])

            out_h2, h2 = self.layers['rnn'](x[:, -tr_time:, :], h1)
            out = self.layers['out'](out_h2)
            out_h = torch.cat((out_h1, out_h2), dim=0)

        else:
            if h:
                out_h, h = self.layers['rnn'](x, h)
            else:
                out_h, h = self.layers['rnn'](x)

            out = self.layers['out'](out_h)


        return choose_output(out, out_h, self.output_type)


    def reset_memory_state(self, batch_size):
        '''
        :param batch_size: size of current batch. 
        '''

        h = (
            torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device),

            torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device)
        )

        return h

    def expand_output_layer(self, n_units=2):
        self.layers["out"] = expand_output_layer(self.layers["out"], n_units)

    def get_layers(self):
        return self.layers.values()


class SketchLSTM(nn.Module):
    """
    Layers are described by the following names:
    'rnn' -> recurrent module
    'out' -> linear readout layer

    Use dict(model.layers[layername].named_parameters()) to get {key : value} dict for parameters of layername.
    """

    def __init__(self, input_size, hidden_size, output_size, device,
                num_layers=1, dropout=0., bidirectional=False,
                orthogonal=False):

        super(SketchLSTM, self).__init__()

        self.is_recurrent = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0.
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.device = device
        self.orthogonal = orthogonal

        self.layers = nn.ModuleDict([])

        self.layers.update([
            ['rnn', nn.LSTM(self.input_size, self.hidden_size, \
                    self.num_layers, batch_first=True, dropout=self.dropout, \
                    bidirectional=self.bidirectional) ]
        ])
        if self.orthogonal:
            for _, hh, _, _ in self.layers['rnn'].all_weights:
                # lstm divides hidden matrix into 4 chunks
                # https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
                for j in range(0, hh.size(0), self.hidden_size):
                    nn.init.orthogonal_(hh[j:j+self.hidden_size])

        self.layers.update([
            ['out', nn.Linear(self.directions*self.hidden_size, self.output_size) ]
        ])

        self.layers = self.layers.to(self.device)

    def forward(self, x, lengths=None, h=None):
        '''
        :param x: (batch_size, seq_len, input_size)
        :param h: hidden state of the recurrent module

        :return out: (batch_size, seq_len, directions*hidden_size)
        :return h: hidden state of the recurrent module
        '''

        #x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        if h:
            out_h, h = self.layers['rnn'](x, h)
        else:
            out_h, h = self.layers['rnn'](x)

        #out_h, lengths = pad_packed_sequence(out_h, batch_first=True)
        # select appropriate h
        #h = torch.gather(out_h, 1, lengths.unsqueeze(1).unsqueeze(1).repeat(1,1,out_h.size(-1)))
        #h = h.squeeze(1) # (batch_size, hidden_size)

        out = self.layers['out'](out_h[:, -1, :])

        return out


    def reset_memory_state(self, batch_size):
        '''
        :param batch_size: size of current batch.
        '''

        h = (
            torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device),

            torch.zeros(self.directions*self.num_layers, batch_size, self.hidden_size,
                device=self.device)
        )

        return h

    def expand_output_layer(self, n_units=2):
        self.layers["out"] = expand_output_layer(self.layers["out"], n_units)

    def get_layers(self):
        return self.layers.values()

