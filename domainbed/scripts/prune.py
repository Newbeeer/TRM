# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from operator import itemgetter

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str, default="RotatedMNIST")
parser.add_argument('--algorithm', type=str, default="ERM")
parser.add_argument('--hparams', type=str,
                    help='JSON-serialized hparams dict')
parser.add_argument('--hparams_seed', type=int, default=0,
                    help='Seed for random hparams (0 means "default hparams")')
parser.add_argument('--trial_seed', type=int, default=0,
                    help='Trial number (used for seeding split_dataset and '
                         'random_hparams).')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for everything else')
parser.add_argument('--epochs', type=int, default=10,
                    help='Training epochs')
parser.add_argument('--steps', type=int, default=None,
                    help='Number of steps. Default is dataset-dependent.')
parser.add_argument('--checkpoint_freq', type=int, default=None,
                    help='Checkpoint every N steps. Default is dataset-dependent.')
parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
parser.add_argument('--output_dir', type=str, default="train_output")
parser.add_argument('--holdout_fraction', type=float, default=0.2)
parser.add_argument('--skip_model_save', action='store_true')
parser.add_argument('--select', action='store_true')
parser.add_argument('--descend', action='store_true')
parser.add_argument('--select_fractions', type=float, default=0.99, help='The fractions of selected data')
args = parser.parse_args()
args.step = 0
# If we ever want to implement checkpointing, just persist these values
# every once in a while, and then load them from disk here.
algorithm_dict = torch.load('train_output/model_erm.pkl')['model_dict']

os.makedirs(args.output_dir, exist_ok=True)
sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tTorchvision: {}".format(torchvision.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tPIL: {}".format(PIL.__version__))

print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

if args.hparams_seed == 0:
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args)
else:
    hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                              misc.seed_hash(args.hparams_seed, args.trial_seed))
if args.hparams:
    hparams.update(json.loads(args.hparams))

print('HParams:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if args.dataset in vars(datasets):
    dataset = vars(datasets)[args.dataset](args.data_dir,
                                           args.test_envs, hparams)
else:
    raise NotImplementedError

# Split each env into an 'in-split' and an 'out-split'. We'll train on
# each in-split except the test envs, and evaluate on all splits.
in_splits = []
out_splits = []
for env_i, env in enumerate(dataset):
    out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i))
    if hparams['class_balanced']:
        in_weights = misc.make_weights_for_balanced_classes(in_)
        out_weights = misc.make_weights_for_balanced_classes(out)
    else:
        in_weights, out_weights = None, None
    in_splits.append(in_)
    out_splits.append(out)

for i, (env) in enumerate(in_splits):
    print("Env:{}, dataset len:{}".format(i, len(env)))

# class of env: ImageFolder
train_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'], shuffle=True,
                                             num_workers=dataset.N_WORKERS) for i, (env) in enumerate(in_splits) if
                 i not in args.test_envs]
eval_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'], shuffle=False,
                                             num_workers=dataset.N_WORKERS)
    for env in (in_splits + out_splits)]
eval_weights = [None for _ in (in_splits + out_splits)]
eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]

algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                            len(dataset) - len(args.test_envs), hparams)
exotic_classifiers = algorithms.EC(dataset.input_shape, dataset.num_classes,
                                   len(dataset) - len(args.test_envs), hparams)
if algorithm_dict is not None:
    algorithm.load_state_dict(algorithm_dict)

algorithm.to(device)
exotic_classifiers.to(device)
steps_per_epoch = min([len(env) / hparams['batch_size'] for env in in_splits])
print("steps per epoch:", steps_per_epoch)
checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
last_results_keys = None


def main(epoch):
    global last_results_keys
    checkpoint_vals = collections.defaultdict(lambda: [])
    train_minibatches_iterator = zip(*train_loaders)

    for batch in train_minibatches_iterator:
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in batch]
        step_vals = algorithm.update(minibatches_device)
        #step_vals, feature = algorithm.update_mi(exotic_classifiers.classifiers, minibatches_device)
        #exotic_classifiers.update_ec(minibatches_device, feature)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        args.step += 1
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if args.step % checkpoint_freq == 0:
            results = {
                'step': args.step,
                'epoch': epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                #print('name:{}, dataset size:{}'.format(name, len(loader.dataset)))
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name + '_acc'] = acc

            results_keys = sorted(results.keys())
            misc.print_row(results_keys, colwidth=12)
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

def eval(algorithm):

    results = dict()
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals:
        # print('name:{}, dataset size:{}'.format(name, len(loader.dataset)))
        acc = misc.accuracy(algorithm, loader, weights, device)
        results[name + '_acc'] = acc

    results_keys = sorted(results.keys())
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([results[key] for key in results_keys],
                   colwidth=12)

if __name__ == "__main__":


    eval(algorithm)

