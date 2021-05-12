import argparse
import os
import copy
import yaml
import re
from types import SimpleNamespace
from shutil import copyfile
from ..experiments.utils import compute_average_intermediate_accuracy
from sklearn.model_selection import ParameterGrid


def set_gpus(num_gpus):
    try:
        import gpustat
    except ImportError:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        selected = []

        stats = gpustat.GPUStatCollection.new_query()

        for i in range(num_gpus):

            ids_mem = [res for res in map(lambda gpu: (int(gpu.entry['index']),
                                          float(gpu.entry['memory.used']) /\
                                          float(gpu.entry['memory.total'])),
                                      stats) if str(res[0]) not in selected]

            if len(ids_mem) == 0:
                # No more gpus available
                break

            best = min(ids_mem, key=lambda x: x[1])
            bestGPU, bestMem = best[0], best[1]
            # print(f"{i}-th best GPU is {bestGPU} with mem {bestMem}")
            selected.append(str(bestGPU))

        print("Setting GPUs to: {}".format(",".join(selected)))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected)
    except BaseException as e:
        print("GPU not available: " + str(e))


def parse_config(config_file):

    # fix to enable scientific notation
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        configs = yaml.load(f, Loader=loader)
    configs['config_file'] = config_file
    args = SimpleNamespace()
    for k,v in configs.items():
        args.__dict__[k] = v

    return args


def create_grid(args):
    """
    Create grid search by returning a list of args
    """
    if args.grid_file == '':
        raise ValueError('Grid file needed.')
    
    final_grid = []
    grid_conf = parse_config(args.grid_file)
    del grid_conf.config_file
    grid = ParameterGrid(vars(grid_conf))
    for el in grid:
        conf = copy.deepcopy(args)
        for k,v in el.items():
            conf.__dict__[k] = v
        final_grid.append(conf)

    return final_grid


def get_best_config(result_folder):
    """
    Select winning config, copy its yaml file into result_folder
    and returns parsed winning args
    """

    ids = [str(el) for el in range(10)]
    dirs = [el for el in os.listdir(result_folder) \
            if os.path.isdir(os.path.join(result_folder, el)) \
            and el.startswith('VAL') \
            and el[-1] in ids]

    best_dir = None
    best_acc = 0
    for dir_path in dirs:
        _, acc, _, _ = compute_average_intermediate_accuracy(os.path.join(result_folder, dir_path))
        if acc > best_acc:
            best_dir = dir_path
            best_acc = acc

    assert best_dir is not None, "Error in retrieving best accuracy"

    copyfile(os.path.join(result_folder, best_dir, 'config_file.yaml'),
             os.path.join(result_folder, 'winner_config.yaml'))
    
    best_config = parse_config(os.path.join(result_folder, 'winner_config.yaml'))

    return best_config


def basic_argparse(parser=None, onemodel=True):

    if parser is None:
        parser = argparse.ArgumentParser()

    # TRAINING
    parser.add_argument('--epochs', type=int, help='epochs to train.')
    if onemodel:
        parser.add_argument('--models', type=str, help='modelname to train')
    else:
        parser.add_argument('--models', nargs='+', type=str, help='modelname to train')
    parser.add_argument('--result_folder', type=str, help='folder in which to save experiment results. Created if not exists.')
    parser.add_argument('--dataroot', type=str, default='/data/cossu', help='folder in which datasets are stored.')

    parser.add_argument('--config_file', type=str, default='', help='path to config file from which to parse args')
    parser.add_argument('--grid_file', type=str, default='', help='path to grid search file from which to create grid')

    # TASK PARAMETERS
    parser.add_argument('--n_tasks', type=int, default=5, help='Task to train.')
    parser.add_argument('--n_tasks_val', type=int, default=3, help='Task to do validation on.')
    parser.add_argument('--n_tasks_test', type=int, default=3, help='Task to do assessment on.')
    parser.add_argument('--output_size', type=int, default=10, help='model output size')
    parser.add_argument('--input_size', type=int, default=1, help='model input size')
    parser.add_argument('--pixel_in_input', type=int, default=1, help='number of pixels to take as last dimension.')
    parser.add_argument('--max_label_value', type=int, default=10, help='Max value for label.')

    # OPTIMIZER
    parser.add_argument('--weight_decay', type=float, default=0, help='optimizer hyperparameter')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='optimizer hyperparameter')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='Value to clip gradient norm.')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD instead of Adam.')

    # EXTRAS
    parser.add_argument('--multitask', action="store_true", help='Multitask learning, all tasks at once.')
    parser.add_argument('--multihead', action="store_true", help='Use task id information at training and test time.')

    parser.add_argument('--not_test', action="store_true", help='disable final test')
    parser.add_argument('--not_intermediate_test', action="store_true", help='disable final test')
    parser.add_argument('--monitor', action="store_true", help='Monitor with tensorboard.')
    parser.add_argument('--save', action="store_true", help='save models')
    parser.add_argument('--load', action="store_true", help='load models')
    parser.add_argument('--cuda', action="store_true", help='use gpu')

    return parser


def add_model_parser(modelnames=['rnn', 'lstm', 'mlp'], parser=None):

    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--expand_output', type=int, default=0, help='Expand output layer dynamically.')

    if 'rnn' or 'lstm' in modelnames:
        parser.add_argument('--hidden_size_rnn', type=int, default=128, help='units of RNN')
        parser.add_argument('--layers_rnn', type=int, default=1, help='layers of RNN')

   
    if 'rnn' or 'lstm' in modelnames:
        parser.add_argument('--orthogonal', action="store_true", help='Use orthogonal recurrent matrixes')

    if 'mlp' in modelnames:
        parser.add_argument('--hidden_sizes_mlp', nargs='+', type=int, default=[128], help='layers of MLP')
        parser.add_argument('--relu_mlp', action="store_true", help='use relu instead of tanh for MLP')

    return parser

def add_cl_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # MNIST
    parser.add_argument('--split', action="store_true", help='Use split MNIST.')
    parser.add_argument('--sequential', action="store_true", help='Do not permute MNIST.')
    
    # SPEECH WORDS
    parser.add_argument('--classes_per_task', type=int, default=2, help='How many classes per tasks (total = 30)')
    
    # EWC / MAS
    parser.add_argument('--ewc_lambda', type=float, default=0., help='Use EWC.')
    parser.add_argument('--mas_lambda', type=float, default=0., help='Use MAS.')
    parser.add_argument('--truncated_time', type=int, default=0, help='Truncated time when computing importance gradient in EWC or MAS')

    # LwF
    parser.add_argument('--lwf', action="store_true", help='Use LWF.')
    parser.add_argument('--lwf_temp', type=float, default=1, help='Set LWF softmax temperature.')
    parser.add_argument('--lwf_alpha', nargs='+', type=float, help='Set LWF alpha.')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='how many warmup epochs for lwf')
    
    # GEM / AGEM
    parser.add_argument('--agem', action="store_true", help='Use A-GEM.')
    parser.add_argument('--agem_sample_size', default=0, type=int, help='How many patterns to take from memory to compute gradient')
    parser.add_argument('--gem', action="store_true", help='Use GEM.')
    parser.add_argument('--gem_patterns_per_step', default=0, type=int, help='How many patterns per step to save in replay memory')
    parser.add_argument('--gem_memory_strength', default=0, type=int, help='offset to add to the projection direction in order to favour backward transfer (gamma in original paper)')
    parser.add_argument('--task_vector_at_test', action="store_true", help='Use task vectors also at test time.')

    # REHEARSAL
    parser.add_argument('--rehe_patterns', type=int, default=0, help='Number of rehearsal patterns per class.')
    parser.add_argument('--patterns_per_class_per_batch', type=int, default=0, help='Add pattern to each minibatch instead of concatenating to the entire dataloader')

    return parser
