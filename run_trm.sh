#!/usr/bin/env bash

gpu=$1
bias=$2

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 3 --bias ${bias}  --shift 0 --test_val

 CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 2 --bias ${bias}  --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 4 --bias ${bias}  --shift 0 --test_val

 CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 5 --bias ${bias}  --shift 0 --test_val

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
# --dataset SceneCOCO --test_envs 0 --trial_seed 4 --bias ${bias}  --shift 2 --test_val
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM \
# --dataset SceneCOCO --test_envs 0 --trial_seed 5 --bias ${bias}  --shift 2 --test_val







