import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
sns.set_palette('dark')
import pandas as pd
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.size': 12})
markers = ["o","v","^","<",">","8","s","p","P","*", ".", "h","H","+","x","X","D","d"]
linestyles = ['--', '-', '-.', ':']

def plot_learning_curves(models, result_folder, additional_metrics=['acc'], title=True, filename='training_results.csv'):
    '''
    :param models: list of modelnames to be used for plots.
    '''

    if isinstance(models, str):
        models = [models]

    with open(os.path.join(result_folder, filename), 'r') as f:
        data_csv = pd.read_csv(f)
        data_csv['task_id'] = data_csv['task_id'].astype('category')

        for modelname in models:
            data_model = data_csv[data_csv['model'] == modelname]

            for metric_type in ['loss'] + additional_metrics:
                data_plot = pd.melt(
                data_model[['epoch', 'task_id', 'train_'+metric_type, 'validation_'+metric_type]], \
                    id_vars=['epoch', 'task_id'], value_vars=['train_'+metric_type, 'validation_'+metric_type],
                    value_name=metric_type)

                data_plot = data_plot[data_plot['epoch'] > 0]

                rp = sns.relplot(
                    x='epoch', kind='line', y=metric_type, hue='task_id', sort=False,
                    legend='full', style='variable', markers=True,
                    data=data_plot)

                rp.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                if title:
                    plt.subplots_adjust(top=0.9)
                    plt.title(f"{modelname} {metric_type}")

                rp.fig.savefig(os.path.join(result_folder, f"{modelname}_{metric_type}.png"))

    return data_plot


def create_writer(folder):
    '''
    Create Tensorboard writer
    '''

    return SummaryWriter(os.path.join(folder, 'tensorboard'))


def plot_importance(writer, modelname, importance, task_id, epoch=0):
    for paramname, imp in importance:
        if len(imp.size()) == 1: # bias
            writer.add_image(f"{modelname}-{paramname}_importance/{task_id}", imp.unsqueeze(0).cpu().data, epoch, dataformats='HW')
        else:
            writer.add_image(f"{modelname}-{paramname}_importance/{task_id}", imp.cpu().view(imp.size(0),-1).data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-{paramname}_importance_hist/{task_id}", imp.cpu().view(-1).data, epoch)


def plot_gradients(writer, modelname, model, task_id, epoch=0):
    for paramname, grad_matrix in model.named_parameters():
        if len(grad_matrix.size()) == 1: # bias
            writer.add_image(f"{modelname}-{paramname}/{task_id}_grad", grad_matrix.unsqueeze(0).cpu().data, epoch, dataformats='HW')
        else: # weights
            writer.add_image(f"{modelname}-{paramname}/{task_id}_grad", grad_matrix.cpu().view(grad_matrix.size(0),-1).data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-{paramname}_grad_hist/{task_id}", grad_matrix.cpu().view(-1).data, epoch)


def plot_weights(writer, modelname, model, task_id, epoch=0):
    for paramname, weight_matrix in model.named_parameters():
        if len(weight_matrix.size()) == 1: # bias
            writer.add_image(f"{modelname}-{paramname}/{task_id}", weight_matrix.unsqueeze(0).cpu().data, epoch, dataformats='HW')
        else: # weights
            writer.add_image(f"{modelname}-{paramname}/{task_id}", weight_matrix.cpu().view(weight_matrix.size(0),-1).data, epoch, dataformats='HW')
        try:
            writer.add_histogram(f"{modelname}-{paramname}_hist/{task_id}", weight_matrix.cpu().view(-1).data, epoch)
        except ValueError:
            print(modelname)
            print(paramname)
            print(weight_matrix.size())
            print(weight_matrix)
            raise ValueError


def plot_activations(writer, modelname, activations, task_id, epoch=0):
    """
    :param activations: list of (hidden_size)
                 or (T, hidden_size) or (batch, T, hidden_size) tensors
    """

    for i, activation in enumerate(activations):

        if len(activation.size()) == 3:
            activation = activation.mean(0)

        if len(activation.size()) == 1:
            activation = activation.unsqueeze(0)

        writer.add_image(f"{modelname}-activation{i}/{task_id}", activation.cpu().data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-activation{i}_hist/{task_id}", activation.cpu().view(-1).data, epoch)


def plot_importance_units(writer, modelname, importances, task_id, epoch=0):
    """
    :param importances: list of (hidden_size)
                 or (T, hidden_size) or (batch, T, hidden_size) tensors
    """
    
    for i, importance in enumerate(importances):

        if len(importance.size()) == 3:
            importance = importance.mean(0)

        if len(importance.size()) == 1:
            importance = importance.unsqueeze(0)

        writer.add_image(f"{modelname}-importance{i}/{task_id}", importance.cpu().data, epoch, dataformats='HW')
        writer.add_histogram(f"{modelname}-importance{i}_hist/{task_id}", importance.cpu().view(-1).data, epoch)


def get_matrix_from_modelname(model, modelname):
    if modelname == 'mlp':
        label = 'i2h'
        weight_matrix = model.layers[label].weight.data
    elif modelname == 'rnn':
        label = 'rnn'
        weight_matrix = model.layers[label].weight_hh_l0.data
    elif modelname == 'lstm':
        label = 'rnn'
        weight_matrix = model.layers[label].weight_hh_l0.data
    
    return weight_matrix, label

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def paired_plot(dest_filename, val_avgs, test_avgs, TITLE=None, DIST=20, INTER_DIST=5):
    """
    :param dest_filename: absolute filename path to save the paired plot in .png format
    :param val_avgs: dict containing, for each modelname as key, the average accuracy over tasks
                    after training on each specific task (left point in plot)
    :param test_avgs: dict containing, for each modelname as key, the average accuracy over tasks
                    after training on last task (right point in plot)
    """

    n_tasks = val_avgs[list(val_avgs.keys())[0]].shape[0]
    cmap = get_cmap(n_tasks, 'Set1')

    xcoords = list(range(8, len(val_avgs.keys())*DIST+1, DIST))
    plt.figure()
    plt.hlines(10, -3, xcoords[-1]+DIST, colors='gray', linewidth=2, linestyle=':', zorder=1, label='random')

    # plot pairs of points
    for i, model in enumerate(val_avgs.keys()):
        for task_id in range(val_avgs[model].shape[0]):
            plt.plot([xcoords[i]-INTER_DIST, xcoords[i]+INTER_DIST], [val_avgs[model][task_id], test_avgs[model][task_id]], c='k', linewidth=0.5, zorder=1)
            plt.scatter([xcoords[i]-INTER_DIST, xcoords[i]+INTER_DIST], [val_avgs[model][task_id], test_avgs[model][task_id]], marker=markers[task_id], s=40, c=[cmap(task_id)], label='T'+str(task_id+1), zorder=2)

    # plot mean vector
    for i, model in enumerate(val_avgs.keys()):
        left_mean = np.mean(val_avgs[model])
        right_mean = np.mean(test_avgs[model])
        plt.plot([xcoords[i]-INTER_DIST, xcoords[i]+INTER_DIST], [left_mean, right_mean], c='red', ls='--', linewidth=0.5, zorder=1)
        plt.scatter([xcoords[i]-INTER_DIST, xcoords[i]+INTER_DIST], [left_mean, right_mean], marker='*', s=40, c='darkred', label='mean', zorder=2)


    # remove duplicated legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.ylabel('Accuracy')
    plt.ylim(-3,105)
    plt.xlim(0,xcoords[-1]+DIST)
    plt.xticks(xcoords, [el for el in val_avgs.keys()])
    plt.grid(True)
    if TITLE is not None:
        plt.title(TITLE)
    if not dest_filename.endswith('.png'):
        dest_filename = dest_filename + '.png'
    plt.savefig(dest_filename)
