import numpy as np
import logging
import os
import sys
import csv
import yaml
from collections import defaultdict
from functools import partial

class LogMetric():

    def __init__(self,
        n_tasks,
        result_folder,
        eval_metric_name='acc',
        intermediate_results_file='intermediate_results.csv'):

        self.n_tasks = n_tasks
        self.result_folder = result_folder
        self.intermediate_results_file = intermediate_results_file
        self.eval_metric_name = eval_metric_name

        # metrics averaged over all previous tasks
        # modelname -> metric -> np.array (n_task, n_task)
        self.intermediate_metrics = defaultdict(lambda: defaultdict(partial(np.zeros, (self.n_tasks,self.n_tasks))))

        # metrics averaged over all epochs for a single task 
        # modelname -> mode -> metric -> task -> list
        self.metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        # running averages within epoch and for intermediate tests
        self.averages = defaultdict(lambda: defaultdict(float))

    def update_averages(self, modelname, l, ev_m=None):
        self.averages[modelname]['loss'] += l
        if ev_m is not None:
            self.averages[modelname][self.eval_metric_name] += ev_m
        
    
    def update_metrics(self, modelname, mode, task_id, l=None, ev_m=None, num_batches=0, reset_averages=False):
        '''
        :param mode: 'train' or 'val' for training metrics or validation metrics
        :param num_batches: number of mini batches over which compute the average.
        '''

        assert( (num_batches > 0) or (l is not None) )

        if num_batches > 0:
            l = self.averages[modelname]['loss'] / float(num_batches)
            if self.eval_metric_name is not None:
                ev_m = self.averages[modelname][self.eval_metric_name] / float(num_batches)

        self.metrics[modelname][mode]['loss'][task_id].append(l)
        if self.eval_metric_name is not None:
            self.metrics[modelname][mode][self.eval_metric_name][task_id].append(ev_m)

        if reset_averages:
            self.reset_averagess(modelname)
    

    def update_intermediate_metrics(self, modelname, num_batches, training_task, intermediate_task, reset_averages=True):

        self.intermediate_metrics[modelname]['loss'][intermediate_task, training_task] = \
            self.averages[modelname]['loss'] / float(num_batches)

        if self.eval_metric_name is not None:
            self.intermediate_metrics[modelname][self.eval_metric_name][intermediate_task, training_task] = \
                 self.averages[modelname][self.eval_metric_name] / float(num_batches)

        if reset_averages:
            self.reset_averagess(modelname)


    def reset_averagess(self, modelname):
        self.averages[modelname]['loss'] = 0.
        if self.eval_metric_name is not None:
            self.averages[modelname][self.eval_metric_name] = 0.


    def get_metric(self, modelname, mode, metricname, task_id, only_last=True):
        '''
        :param mode: 'train' or 'val' for training metrics or validation metrics
        :param metricname: 'loss' or eval metric name
        '''

        m = self.metrics[modelname][mode][metricname][task_id]

        if only_last:
            return m[-1]
        else:
            return m

    def save_intermediate_metrics(self):
        '''
        Save intermediate dictionaries to file.
        '''

        with open(os.path.join(self.result_folder, self.intermediate_results_file), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            if self.eval_metric_name is not None:
                csvwriter.writerow(['model', 'intermediate_test_task', 'training_task', self.eval_metric_name, 'loss'])
            else:
                csvwriter.writerow(['model', 'intermediate_test_task', 'training_task', 'loss'])

            for modelname, vs in self.intermediate_metrics.items():
                # every value in position i,j is the loss/eval_metric_name on task i after training on task j
                losses = vs['loss']

                if self.eval_metric_name is not None:
                    ev_metrics = vs[self.eval_metric_name]

                for training_task in range(losses.shape[0]):
                    for intermediate_task in range(0, training_task+1):
                        cur_l = losses[intermediate_task, training_task]
                        if self.eval_metric_name is not None:
                            cur_ev_m = ev_metrics[intermediate_task, training_task]
                            csvwriter.writerow([modelname, intermediate_task, training_task, cur_ev_m, cur_l])
                        else:
                            csvwriter.writerow([modelname, intermediate_task, training_task, cur_l])


def create_logger(base_folder, log_file='training_results.csv'):
    '''
    logger.info print on console only
    logger.warning print on console and file (results.log)
    '''

    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger(base_folder)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=os.path.join(base_folder, log_file), mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def write_configuration(args, folder):
    '''
    Write the input argument passed to the script to a file
    '''

    with open(os.path.join(folder, 'config_file.yaml'), 'w') as f:
        yaml.dump(dict(vars(args)), f)
