# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
from torchvision import transforms
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from domainbed.celeba_dataset import CelebA_group
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str, default="ColoredMNIST")
parser.add_argument('--algorithm', type=str, default="ERM")
parser.add_argument('--shift', type=int, default=0,
                    help="0:label-correlated; 1: label-uncorrelated; 2: combine.")
parser.add_argument('--holdout_fraction', type=float, default=0.2)
parser.add_argument('--opt', type=str, default="SGD")
parser.add_argument('--hparams', type=str,
                    help='JSON-serialized hparams dict')
parser.add_argument('--hparams_seed', type=int, default=0,
                    help='Seed for random hparams (0 means "default hparams")')
parser.add_argument('--trial_seed', type=int, default=0,
                    help='Trial number (used for seeding split_dataset and '
                         'random_hparams).')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for everything else')
parser.add_argument('--lr', type=float, default=None,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='Training epochs')
parser.add_argument('--steps', type=int, default=None,
                    help='Number of steps. Default is dataset-dependent.')
parser.add_argument('--checkpoint_freq', type=int, default=None,
                    help='Checkpoint every N steps. Default is dataset-dependent.')
parser.add_argument('--output_dir', type=str, default="checkpoint")
parser.add_argument('--save_path', type=str, default="model")
parser.add_argument('--resume_path', default="model", type=str, help='path to resume the checkpoint')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--bias', type=float, default=0.9)
parser.add_argument('--irm_lam', type=float, default=1)
parser.add_argument('--rex_lam', type=float, default=1)
parser.add_argument('--cos_lam', type=float, default=1e-4)
parser.add_argument('--swap_lam', type=float, default=1.)
parser.add_argument('--trm_lam', type=float, default=1.)
parser.add_argument('--chunk', type=int, default=2)
parser.add_argument('--dro_eta', type=float, default=1e-2)
parser.add_argument('--skip_model_save', action='store_true')
parser.add_argument('--resnet18', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--robust', action='store_true')
parser.add_argument('--reweight', action='store_true')
parser.add_argument('-i', '--img_size', type=int, default=64)
parser.add_argument('--test_val', action='store_true', help="test-domain validation set")
parser.add_argument('--class_balanced', action='store_true')
args = parser.parse_args()
args.step = 0
args.test_envs = []
# If we ever want to implement checkpointing, just persist these values
# every once in a while, and then load them from disk here.
algorithm_dict = None

os.makedirs(args.output_dir, exist_ok=True)
sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

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

if args.lr is not None:
    hparams['lr'] = args.lr

print('HParams:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(args.trial_seed)
np.random.seed(args.trial_seed)
torch.manual_seed(args.trial_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if args.dataset in vars(datasets):
    dataset = vars(datasets)[args.dataset](args.data_dir, args.bias, args.test_envs, hparams, image_size=args.img_size)
else:
    raise NotImplementedError

# Split each env into an 'in-split' and an 'out-split'. We'll train on
# each in-split except the test envs, and evaluate on all splits.
in_splits = []
val_splits = []
test_splits = []
for env_i, env in enumerate(dataset):
    val, test, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i), val_split=True)
    if hparams['class_balanced']:
        in_weights = misc.make_weights_for_balanced_classes(in_)
        out_weights = misc.make_weights_for_balanced_classes(val)
    else:
        in_weights, out_weights = None, None
    in_splits.append(in_)
    val_splits.append(val)
    test_splits.append(test)

for i, (env) in enumerate(in_splits):
    print("train split: Env:{}, dataset len:{}".format(i, len(env)))
    print("test split: Env:{}, dataset len:{}".format(i, len(test_splits[i])))
    print("val split: Env:{}, dataset len:{}".format(i, len(val_splits[i])))
if not args.reweight:
    image_size = args.img_size
    if image_size == 224:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif image_size == 64:
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.CenterCrop(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    dataset_erm = CelebA_group('/home/ylxu/domainbed', 'male_blond', transform=transform, label=0)
    dataset_erm.filename = list(in_splits[0].underlying_dataset.filename[in_splits[0].keys])
    dataset_erm.label = in_splits[0].underlying_dataset.label[in_splits[0].keys]
    print("length:", len(dataset_erm.filename))
    for i, (env) in enumerate(in_splits):
        if i == 0:
            continue
        print("length:", len(env.underlying_dataset.filename[env.keys]))
        dataset_erm.filename += list(env.underlying_dataset.filename[env.keys])
        dataset_erm.label = torch.cat((dataset_erm.label,env.underlying_dataset.label[env.keys]), dim=0)
    if args.dataset == 'Celeba':
        hparams['batch_size'] = 64
    train_loaders = [torch.utils.data.DataLoader(dataset=dataset_erm, batch_size=hparams['batch_size'], shuffle=True,
                                             num_workers=dataset.N_WORKERS)]

    print("Env 0 (ERM), dataset len:{}".format(len(dataset_erm)))
else:
    # class of env: ImageFolder
    train_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'], shuffle=True,
                                             num_workers=dataset.N_WORKERS) for i, (env) in enumerate(in_splits)]

eval_loaders = [torch.utils.data.DataLoader(dataset=env, batch_size=hparams['batch_size'], shuffle=False,
                                             num_workers=dataset.N_WORKERS)
    for env in (in_splits + val_splits + test_splits)]


eval_weights = [None for _ in (in_splits + val_splits + test_splits)]
eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
eval_loader_names += ['env{}_val'.format(i) for i in range(len(val_splits))]
eval_loader_names += ['env{}_test'.format(i) for i in range(len(test_splits))]

algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                            len(dataset) - len(args.test_envs), hparams)
if algorithm_dict is not None:
    algorithm.load_state_dict(algorithm_dict)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.resume_path}.pth')
    algorithm.load_state_dict(checkpoint['model_dict'], strict=False)


if args.parallel:
    algorithm = torch.nn.DataParallel(algorithm, device_ids=[0,1])
algorithm.cuda()
if args.dataset != 'PACS':
    steps_per_epoch = min([len(env) / hparams['batch_size'] for env in in_splits])
    print("steps per epoch:", steps_per_epoch)
    args.epochs = int(2000. / steps_per_epoch)
checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
last_results_keys = None
best_acc_in = 0.
best_acc_out = 0.
best_acc_val_worse = 0.
collect_dict = collections.defaultdict(lambda: [])
def main(epoch):
    global last_results_keys
    global best_acc_out
    global best_acc_in
    global collect_dict
    checkpoint_vals = collections.defaultdict(lambda: [])
    train_minibatches_iterator = zip(*train_loaders)
    for batch in train_minibatches_iterator:
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in batch]

        step_vals = algorithm.update(minibatches_device)
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
                acc = misc.accuracy(algorithm, loader, weights, device, args.parallel)
                results[name + '_acc'] = acc

            results_keys = sorted(results.keys())
            misc.print_row(results_keys, colwidth=12)
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)


            #########
            worse_acc_val = 100.
            worse_acc_test = 100.
            cnt_val = 0.
            cnt_test = 0.
            avg_acc_val = 0.
            avg_acc_test = 0.
            for i in range(len(val_splits)):

                name_val = 'env{}_val_acc'.format(i)
                worse_acc_val = min(worse_acc_val, results[name_val])
                avg_acc_val += results[name_val] * len(val_splits[i])
                cnt_val += len(val_splits[i])

                name_test = 'env{}_test_acc'.format(i)
                worse_acc_test = min(worse_acc_test, results[name_test])
                avg_acc_test += results[name_test] * len(test_splits[i])
                cnt_test += len(test_splits[i])
            avg_acc_val /= cnt_val
            avg_acc_test /= cnt_test
            global best_acc_val_worse
            if worse_acc_val > best_acc_val_worse:
                best_acc_val_worse = worse_acc_val
                print("-------------------------------------")
                print(f"epoch {epoch}, val worse:{best_acc_val_worse}, val avg:{avg_acc_val},"
                      f" test worse:{worse_acc_test}, test avg:{avg_acc_test}")
                print("-------------------------------------")

            #
            # if args.test_val:
            #     name_in = 'env{}_in_acc'.format(args.test_envs[0])
            #     name_out = 'env{}_out_acc'.format(args.test_envs[0])
            #     val_acc = results[name_in]
            #     test_acc = results[name_out]
            # collect_dict['acc'].append(test_acc)
            # if val_acc > best_acc_in:
            #     best_acc_in = val_acc
            #     best_acc_out = test_acc
            #     path = os.path.join(args.output_dir, f"{args.algorithm}_bias_{args.bias}_seed_{args.trial_seed}_{args.dataset}.pth")
            #     # print("New best checkpoint with acc:{}".format(best_acc_out), "- save to ", path)
            #     print(f"epoch:{epoch}, Val acc:{best_acc_in:.4f}, Test acc:{best_acc_out:.4f}")
            #     # if not args.skip_model_save:
            #     #     save_dict = {
            #     #         "args": vars(args),
            #     #         "model_input_shape": dataset.input_shape,
            #     #         "model_num_classes": dataset.num_classes,
            #     #         "model_num_domains": len(dataset) - len(args.test_envs),
            #     #         "model_hparams": hparams,
            #     #         "model_dict": algorithm.state_dict()
            #     #     }
            #     #     print("save at...", path)
            #     #     torch.save(save_dict, path)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')


if __name__ == "__main__":

    import time
    avg_time = 0
    if args.algorithm == 'ERM':
        args.epochs = 2
    print("Training epoch:", args.epochs)
    for epoch in range(args.epochs):
        start_time = time.time()
        main(epoch)
        end_time = time.time()
        avg_time += (end_time - start_time)
    print("Average time for an epoch:", avg_time / args.epochs)
    print(f"seed:{args.trial_seed},alg:{args.algorithm},bias:{args.bias}, Best val acc:{best_acc_in:.4f}, Best test acc:{best_acc_out:.4f}")
    # np.save('loss_'+str(args.algorithm), collect_dict['loss'])
    # if args.algorithm != 'IRM':
    #     np.save('trm_'+str(args.algorithm), collect_dict['trm'])
    # if args.algorithm != 'TRM':
    #     np.save('penalty_' + str(args.algorithm), collect_dict['penalty'])
    #
    # np.save('acc_' + str(args.algorithm), collect_dict['acc'])