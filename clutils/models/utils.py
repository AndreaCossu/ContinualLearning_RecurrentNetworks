import torch
import torch.nn as nn


def expand_output_layer(layer, n_units):
    """
    Expand layer wiht n_units more.
    layer can be either a Linear layer or a (weight, bias) tuple of Parameters.

    Return a new torch.nn.Linear
    """

    if isinstance(layer, tuple):
        weight = layer[0]
        bias = layer[1]
    elif isinstance(layer, nn.Linear):
        weight = layer.weight
        bias = layer.bias
    else:
        raise ValueError(f"layer must be torch.nn.Linear or tuple of Parameters. Got {type(layer)}.")

    with torch.no_grad():
        # recompute output size
        old_output_size = weight.size(0)
        hidden_size = weight.size(1)
        new_output_size = old_output_size + n_units

        # copy old output layer into new one
        new_layer = nn.Linear(hidden_size, new_output_size, bias=True).to(weight.device)
        new_layer.weight.data[:old_output_size, :] = weight.clone()
        new_layer.bias.data[:old_output_size] = bias.clone()

        return new_layer


def sequence_to_flat(x):
    n_dims = len(x.size())

    if n_dims > 2:
        return x.view(x.size(0), -1)

    return x


def init_weights(model, initw=None):
    if initw is None:
        def initw(m):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)
    model.apply(initw)


def zero_weight(m):
    nn.init.constant_(m.weight, 0.)
    nn.init.constant_(m.bias, 0.)


def compute_conv_out_shape(Win, Hin, padding, dilation, kernel_size, stride):
    return (
        int(((Win + 2*padding - dilation * (kernel_size - 1) - 1) / stride) + 1),
        int(((Hin + 2*padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    )

def compute_conv_out_shape_1d(window_size, padding, dilation, kernel_size, stride):
    return int(((window_size + 2*padding - dilation * (kernel_size - 1) - 1) / stride) + 1)