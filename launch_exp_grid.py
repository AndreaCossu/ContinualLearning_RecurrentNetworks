import argparse
import copy
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"  # This is CRUCIAL to avoid bottlenecks when running experiments in parallel. DO NOT REMOVE IT
import ray
import torch
from clutils.extras import basic_argparse, add_model_parser, add_cl_parser, parse_config
from clutils import monitors
from mnist import mnist_exp, mnist_multitask_exp
from speech_words import words_exp, words_multitask_exp
from quickdraw import quickdraw_exp, quickdraw_multitask_exp
from bbbmnist import bbbmnist_exp
from clutils.extras import set_gpus, get_best_config, create_grid

parser = basic_argparse()
parser = add_model_parser(parser=parser)
parser = add_cl_parser(parser=parser)

parser.add_argument('--exp_type', type=str, choices=['mnist'], help='Type of experiment to execute')
parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--max_cpus', type=int, default=1, help='Maximum number of CPUs')
parser.add_argument('--max_gpus', type=int, default=1, help='Maximum number of GPUs')
parser.add_argument('--gpus_per_job', type=int, default=1, help='Maximum number of GPUs per job')
parser.add_argument('--cpus_per_job', type=int, default=1, help='Maximum number of CPUs per job')
parser.add_argument('--not_ray', action="store_true", help='Do not use ray')

args = parser.parse_args()

if args.config_file == '':
    raise ValueError('Grid search must be executed with a config file.')


torch.set_num_threads(args.cpus_per_job)

args = parse_config(args.config_file)
if args.grid_file == '':
    raise ValueError('Grid search must be executed with a grid file.')

assert (args.n_tasks_val == -1 and args.n_tasks_test == -1) or \
       (args.n_tasks_val > 0 and args.n_tasks_test > 0), \
       "n_tasks_val and n_tasks_test must either be both -1 or both > 0."
assert hasattr(args, 'tasks_list') or args.n_tasks_val > 0, "Cannot use n_tasks_val or n_tasks_test -1 with no tasks list."

if hasattr(args, 'tasks_list') and args.n_tasks_val > 0:
    assert(args.n_tasks_val + args.n_tasks_test <= len(args.tasks_list)), \
        "You specified more tasks for val and test than tasks list."

orig_args = copy.deepcopy(args)
grid_args = create_grid(args)

if args.not_ray:
    raise ValueError('Grid search must be executed with ray.')

try:
    @ray.remote(num_cpus=args.cpus_per_job, num_gpus=args.gpus_per_job)
    def run_exp(argum):
        if args.cuda:
            print(f'Using GPU {ray.get_gpu_ids()}')
        else:
            print('Using CPUs')

        if argum.exp_type == 'mnist':
            if args.multitask:
                mnist_multitask_exp(argum)
            else:
                mnist_exp(argum)
        elif argum.exp_type == 'words':
            if args.multitask:
                words_multitask_exp(argum)
            else:
                words_exp(argum)
        elif argum.exp_type == 'quickdraw':
            if args.multitask:
                quickdraw_multitask_exp(argum)
            else:
                quickdraw_exp(argum)
        else:
            print("Experiment type not recognized.")

    if args.cuda:
        # Execution will be sequential
        torch.set_num_threads(args.max_cpus)
        set_gpus(args.max_gpus)
        ray.init(num_cpus=args.max_cpus, num_gpus=args.max_gpus)
    elif os.environ.get('ip_head') is not None:
        assert os.environ.get('redis_password') is not None, "Missing redis password"
        ray.init(address=os.environ.get('ip_head'), _redis_password=os.environ.get('redis_password'))
        print("Connected to Ray cluster.")
        print(f"Available nodes: {ray.nodes()}")
        args.gpus_per_job = 0
    else:
        torch.set_num_threads(args.max_cpus)
        ray.init(num_cpus=args.max_cpus)
        args.gpus_per_job = 0
        print(f"Started local ray instance.")

    assert ray.is_initialized(), "Error in initializing ray."

    #### START VALIDATION ####

    remaining_ids = []
    for grid_id, curr_args in enumerate(grid_args):
        # create jobs
        curr_args.result_folder = os.path.join(orig_args.result_folder, f'VAL{grid_id}')
        curr_args.model_selection = True

        if hasattr(orig_args, 'tasks_list'): # select validation tasks
            if orig_args.n_tasks_val == -1:
                curr_args.tasks_list = orig_args.tasks_list
            else:
                curr_args.tasks_list = orig_args.tasks_list[:orig_args.n_tasks_val]
            curr_args.n_tasks = len(curr_args.tasks_list)
        else:
            curr_args.n_tasks = orig_args.n_tasks_val

        remaining_ids.append(run_exp.remote(curr_args))
    n_jobs = len(remaining_ids)
    print(f"Total jobs: {n_jobs}")
    print(remaining_ids)

    # wait for jobs
    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
        for result_id in done_ids:
            # There is only one return result by default.
            result = ray.get(result_id)
            n_jobs -= 1
            print(f'Job {result_id} terminated.\nJobs left: {n_jobs}')


    best_args = get_best_config(orig_args.result_folder)

    #### START ASSESSMENT ####
    remaining_ids = []
    times = {}
    for i in range(best_args.num_runs):
        best_args.result_folder = os.path.join(orig_args.result_folder, f'ASSESS{i}')
        best_args.input_size = orig_args.input_size
        best_args.model_selection = False

        if hasattr(best_args, 'tasks_list'):
            if orig_args.n_tasks_test == -1:
                best_args.tasks_list = orig_args.tasks_list
            else:
                best_args.tasks_list = orig_args.tasks_list[orig_args.n_tasks_val:orig_args.n_tasks_val+orig_args.n_tasks_test]
            best_args.n_tasks = len(best_args.tasks_list)
        else:
            best_args.n_tasks = orig_args.n_tasks_test

        remaining_ids.append(run_exp.remote(best_args))
        times[remaining_ids[-1]] = time.time()

    n_jobs = len(remaining_ids)
    print(f"Total jobs: {n_jobs}")
    print(remaining_ids)

    # wait for jobs
    while remaining_ids:
        done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
        for result_id in done_ids:
            # There is only one return result by default.
            result = ray.get(result_id)
            times[result_id] = time.time() - times[result_id]
            n_jobs -= 1

            print(f'Job {result_id} terminated.\nJobs left: {n_jobs}')

    average_time = sum(list(times.values())) / float(len(times.keys()))
    with open(os.path.join(orig_args.result_folder, 'other_metrics.txt'), 'a') as f:
        f.write(f'time,{average_time:.0f}\n')

finally:
    print('Shutting down ray...')
    ray.shutdown()
    print('Ray closed.')
