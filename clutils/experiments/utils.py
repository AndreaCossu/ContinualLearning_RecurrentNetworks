import os
import torch
import pandas as pd
import numpy as np
from ..models import VanillaRNN, LSTM, MLP, ESN, SketchLSTM


def create_result_folder(result_folder, path_save_models='saved_models'):
    '''
    Set plot folder by creating it if it does not exist.
    '''

    result_folder = os.path.expanduser(result_folder)
    os.makedirs(os.path.join(result_folder, path_save_models), exist_ok=True)
    return result_folder


def get_device(cuda):
    '''
    Choose device: cpu or cuda
    '''

    mode = 'cpu'
    if cuda and torch.cuda.is_available():
        mode = 'cuda'
    device = torch.device(mode)

    return device


def save_model(model, modelname, base_folder, path_save_models='saved_models', version=''):
    '''
    :param version: specify version of the model. Usually used to represent the model when trained after task 'version'
    '''

    torch.save(model.state_dict(), os.path.join(
        os.path.expanduser(base_folder), 
        path_save_models, modelname+version+'.pt'))


def load_models(model, modelname, device, base_folder, path_save_models='saved_models', version=''):
    check = torch.load(os.path.join(
        os.path.expanduser(base_folder),
        path_save_models, modelname+version+'.pt'), map_location=device)

    model.load_state_dict(check)

    model.eval()

    return model


def create_models(args, device,
                  quickdraw=False,
                  path_save_models='saved_models', version=''):
    '''
    Create models for CL experiment.

    :param version: string representing version of models to load.
    '''

    models = {}

    if 'rnn' in args.models:
        models['rnn'] = VanillaRNN(args.input_size, args.hidden_size_rnn, args.output_size, device,
            num_layers=args.layers_rnn, orthogonal=args.orthogonal)

    if 'lstm' in args.models:
        if not quickdraw:
            models['lstm'] = LSTM(args.input_size, args.hidden_size_rnn, args.output_size, device,
                              num_layers=args.layers_rnn, orthogonal=args.orthogonal)
        else:
            models['lstm'] = SketchLSTM(args.input_size, args.hidden_size_rnn, args.output_size, device,
                num_layers=args.layers_rnn, orthogonal=args.orthogonal, bidirectional=args.bidirectional)

    if 'mlp' in args.models:
        models['mlp'] = MLP(args.input_size, args.hidden_sizes_mlp, device, output_size=args.output_size, relu=args.relu_mlp)

    if args.load:
        for modelname in args.models:
            models[modelname] = load_models(models[modelname], modelname, device, 
            args.result_folder, path_save_models, version=version)
    
    return models


def create_optimizers(models, lr, wd=0., use_sgd=False):
    '''
    Associate an optimizer to each model
    '''

    optimizers = {}
    optimizer_class = torch.optim.SGD if use_sgd else torch.optim.Adam
    for modelname, model in models.items():
        optimizers[modelname] = optimizer_class(model.parameters(),
            lr=lr, weight_decay=wd)

    return optimizers


def clip_grad(model, max_grad_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def detach(h):
    if isinstance(h, (tuple, list)):
        return tuple([hh.detach() for hh in h])
    else:
        return h.detach()


def compute_average_intermediate_accuracy(folder, intermediate_result_name='intermediate_results.csv'):
    """
    Return average accuracy over all tasks on a specified result folder,
    after training on all tasks.
    """

    cur_file = os.path.join(folder, intermediate_result_name)
    data = pd.read_csv(cur_file)
    data = data[data['training_task'] == data['training_task'].max()] # choose last task
    data = data[['loss', 'acc']].values

    # both are array of 2 elements (loss, acc)
    loss, acc = np.average(data, axis=0) 
    loss_std, acc_std = np.std(data, axis=0)
    
    return loss, acc, loss_std, acc_std

def compute_training_mean_std(
        root, 
        run_foldername='run', 
        training_result_name='training_results.csv'):
    """
    :param root: absolute path to the folder containing all the runs to be averaged
                Each run folder must be called `run_foldername`i
    :param run_foldername: name of the folder of each run. A progressive index i
                            will be appended to it to distinguish one run from another.
    """    

    num_runs = len([el for el in os.listdir(root) if el.startswith(run_foldername)])
    data_gathered = None
    for i in range(num_runs):
        cur_file = os.path.join(root, run_foldername+str(i), training_result_name)
        data = pd.read_csv(cur_file)
        data = data[data['epoch'] == data['epoch'].max()] # choose last epoch
        data = np.expand_dims(data[['train_acc', 'validation_acc', 'train_loss', 'validation_loss']].values, axis=0)
        if data_gathered is None:
            data_gathered = data
        else:
            data_gathered = np.concatenate((data_gathered, data), axis=0)
        
    averages = np.average(data_gathered, axis=0)
    stds = np.std(data_gathered, axis=0)
    
    tr_acc_mean, val_acc_mean, tr_loss_mean, val_loss_mean = averages[:,0], averages[:,1], averages[:,2], averages[:,3]
    tr_acc_std, val_acc_std, tr_loss_std, val_loss_std = stds[:,0], stds[:,1], stds[:,2], stds[:,3]
    
    return (tr_acc_mean, val_acc_mean, tr_loss_mean, val_loss_mean), \
           (tr_acc_std, val_acc_std, tr_loss_std, val_loss_std) 


def compute_intermediate_mean_std(
        root,
        run_foldername='run',
        intermediate_result_name='intermediate_results.csv'
    ):
    """
    :param root: absolute path to the folder containing all the runs to be averaged
                Each run folder must be called `run_foldername`i
    :param run_foldername: name of the folder of each run. A progressive index i
                            will be appended to it to distinguish one run from another.
    """

    num_runs = len([el for el in os.listdir(root) if el.startswith(run_foldername)])
    data_gathered = None
    for i in range(num_runs):
        cur_file = os.path.join(root, run_foldername+str(i), intermediate_result_name)
        data = pd.read_csv(cur_file)
        data = data[data['training_task'] == data['training_task'].max()] # choose last task
        data = np.expand_dims(data[['loss', 'acc']].values, axis=0)
        if data_gathered is None:
            data_gathered = data
        else:
            data_gathered = np.concatenate((data_gathered, data), axis=0)
        
    averages = np.average(data_gathered, axis=0)
    stds = np.std(data_gathered, axis=0)
    
    loss_mean, acc_mean = averages[:,0], averages[:,1]
    loss_std, acc_std = stds[:,0], stds[:,1]
    
    return (loss_mean, acc_mean), (loss_std, acc_std) 
