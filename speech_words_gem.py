from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn import LSTM
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, timing_metrics, forgetting_metrics
from avalanche.training.strategies import GEM
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.logging import CSVLogger
from avalanche.training.plugins import EvaluationPlugin
import json
import torchaudio
from sys import platform
if platform == 'win32' or platform == 'cygwin': # windows backend
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    torchaudio.set_audio_backend("soundfile")
else: # use linux backend by default
    torchaudio.set_audio_backend("sox_io")
from typing import Tuple, Any, Union, Sequence
from torchaudio import transforms
from torch.utils.data import Dataset

CLASS_TO_ID = { 'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5,
                'five': 6, 'four': 7, 'go': 8, 'happy': 9, 'house': 10, 'left': 11,
                'marvel': 12, 'nine': 13, 'no': 14, 'off': 15, 'on': 16, 'one': 17,
                'right': 18, 'seven': 19, 'sheila': 20, 'six': 21, 'stop': 22, 'three': 23,
                'tree': 24, 'two': 25, 'up': 26, 'wow': 27, 'yes': 28, 'zero': 29}


class SSC(Dataset):
        """Synthetic Speech Commands Recognition
        https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset
        This class works with both `Augmented Dataset` and `Augmented Dataset very noisy` version.
        Each audio is preprocessed with Mel Spectrogram. The resulting sequence has the same length
        for all audio files. The length of the sequence depends on the Mel Spectrogram's parameters.

        The test set contains 20% of the entire dataset.

        Args:
            root: folder where audio files are. Audio are grouped in one folder per class.
            classes: list of names of classes to load. None to load all classes. Default to None.
            targets_from_zero: if True, classes will be remapped when necessary to have labels starting from 0
                to n_classes - 1. Only needed when classes is not None.
            split: 'train' or 'test'. Fixed split based on filenames from original dataset.
            n_mels: number of MEL FCSSs feature to create for each audio during preprocessing
            win_length: length of sliding window for Mel Spectrogram in frames
            hop_length: distance between one window and the next in frames
        """
        def __init__(self,
                     root: str = '.data',
                     classes: Union[Sequence[str], None] = None,
                     targets_from_zero: bool = True,
                     split: str = 'train',
                     n_mels: int = 40,
                     win_length: int = 25,
                     hop_length: int = 10,

                     ) -> None:
            super().__init__()

            self.root = root
            assert split == 'train' or split == 'test', "Wrong split for SSC."
            self.split = split
            self.n_mels = n_mels  # can be used as input size for models
            self.sample_rate = 16000  # each audio must have this sample rate

            win_length = int(self.sample_rate / 1000 * win_length)
            hop_length = int(self.sample_rate / 1000 * hop_length)
            self.mel_spectr = transforms.MelSpectrogram(sample_rate=self.sample_rate,
                win_length=win_length, hop_length=hop_length, n_mels=n_mels)

            # remap class labels to the selected classes
            if classes is not None:
                assert all([el in list(CLASS_TO_ID.keys()) for el in classes]), "Wrong class name for SSC."
                self.classes = classes
            else:
                self.classes = list(CLASS_TO_ID.keys())

            if targets_from_zero and classes is not None:
                self.class_to_id = dict(zip(classes, list(range(len(classes)))))
            else:
                self.class_to_id = CLASS_TO_ID

            self.data = None
            self.targets = None
            self._load_data()  # preprocess data and set data and targets

        def _load_data(self) -> None:
            """
            Load all audio files and associate the corresponding target.
            Also split between train and test following a fixed split.
            """

            # load test split containing, for each class
            # the test filenames
            with open("scr_test_split.json", "r") as f:
                test_split_dict = json.load(f)

            data = []
            targets = []
            for classname in self.classes:
                files = [el for el in os.listdir(os.path.join(self.root, classname))
                         if el.endswith('.wav')]

                features = []
                for i, f in enumerate(files):
                    # load appropriate files based on fixed split
                    if self.split == 'test' and f not in test_split_dict[classname]:
                        continue
                    elif self.split == 'train' and f in test_split_dict[classname]:
                        continue

                    audio, sample_rate = torchaudio.load(os.path.join(self.root, classname, f))
                    assert sample_rate == self.sample_rate
                    features.append(self.mel_spectr(audio).permute(0, 2, 1))

                data.append(torch.cat(features, dim=0)) # batch-first sequence
                targets.append(torch.ones(data[-1].size(0)).long() * self.class_to_id[classname])

            self.data = torch.cat(data)
            self.targets = torch.cat(targets)

        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            """

            Args:
                index: numerical index used to select the pattern

            Returns: (audio, target) where
                audio is the preprocessed audio tensor of size [length, n_mels]
                and target is the class index tensor of size []
            """
            return self.data[index], self.targets[index]

        def __len__(self) -> int:
            return len(self.data)



class SequenceClassifier(nn.Module):
    def __init__(self, rnn, hidden_size, output_size):
        super().__init__()
        self.rnn = rnn
        self.classifier = nn.Linear(hidden_size, output_size)
        assert self.rnn.batch_first

    def forward(self, x):
        x = self.rnn(x)
        x = x[0][:, -1]  # last timestep
        return self.classifier(x)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.act = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act(self.input_layer(x))
        x = self.act(self.hidden_layer(x))
        out = self.output_layer(x)
        return out
        
def ssc_gem(args):

    mode = 'cpu'
    if args.cuda and torch.cuda.is_available():
        mode = 'cuda'
    device = torch.device(mode)

    # --- SCENARIO
    input_size = 40

    if 'model_selection' not in vars(args):
        args.model_selection = False

    if args.model_selection:
        classes = ['bed', 'bird', 'cat', 'dog', 'down', 'eight']
        n_exp = 3
    else:
        classes = ['house', 'left', 'marvel', 'nine', 'no', 'off', 'one', 'on', 'right', 'seven', 'sheila',
                   'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
        n_exp = 10

    dataset_train = SSC(args.dataroot, split='train', n_mels=input_size, classes=classes)
    dataset_test = SSC(args.dataroot, split='test', n_mels=input_size, classes=classes)
    scenario = nc_benchmark(dataset_train, dataset_test, n_exp,
                            task_labels=False, seed=1234,
                            fixed_class_order=list(range(len(classes))))

    if args.use_rnn:
        model = SequenceClassifier(
            LSTM(input_size, args.rnn_units, batch_first=True, num_layers=2),
            args.rnn_units, scenario.n_classes)
    else:
        model = MLP(input_size*101, args.mlp_units, scenario.n_classes)

    criterion = CrossEntropyLoss()

    if args.opt_name == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Optimizer name not recognized.")

    # --- STRATEGY
    f = open(os.path.join(args.result_folder, 'text_logger.txt'), 'w')
    text_logger = TextLogger(f)
    csv_logger = CSVLogger(args.result_folder)
    tensorboard_logger = TensorboardLogger(os.path.join(args.result_folder, "tb_data"))

    # same number of classes per exp
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[text_logger, csv_logger, tensorboard_logger])

    cl_strategy = GEM(model, optimizer, criterion, train_mb_size=args.batch_size,
                      train_epochs=args.epochs, device=device,
                      evaluator=eval_plugin, eval_every=1,
                      patterns_per_exp=args.patterns_per_exp, memory_strength=args.strength)
    # --- TRAINING LOOP

    print('Starting experiment...')

    for i, exp in enumerate(scenario.train_stream):
        print("Start training on experience ", exp.current_experience)

        cl_strategy.train(exp, eval_streams=[scenario.test_stream[i]])
        print('Training completed')

        print('Computing accuracy on the whole test set')
        cl_strategy.eval(scenario.test_stream)

    f.close()
    csv_logger.close()


