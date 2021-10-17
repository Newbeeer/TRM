#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM \
 --dataset ColoredMNIST --test_envs 0 --trial_seed 4 --bias 0.9

 CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO \
 --dataset Celeba --test_envs 0 --trial_seed 4 --bias 0.9

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO \
 --dataset Celeba --test_envs 0 --trial_seed 4 --bias 0.9

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM \
 --dataset ColoredMNIST --test_envs 0 --trial_seed 4 --bias 0.9

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
 --dataset ColoredMNIST --test_envs 0 --trial_seed 4 --bias 0.9

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
 --dataset ColoredMNIST --test_envs 0 --trial_seed 4 --bias 0.9

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
 --dataset ColoredMNIST --test_envs 0 --trial_seed 4 --bias 0.9 --iters 100

