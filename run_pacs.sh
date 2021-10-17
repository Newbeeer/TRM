#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train <test_envs> <gpu_num>"
  exit 1
fi

test_env=$1
gpu=$2

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM  --dataset PACS --test_envs ${test_env} --trial_seed 2

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM  --dataset PACS --test_envs ${test_env} --trial_seed 2 --irm_lam 1

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx  --dataset PACS --test_envs ${test_env} --trial_seed 2 --rex_lam 1

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm GroupDRO  --dataset PACS --test_envs ${test_env} --trial_seed 2 --dro_eta 1e-2

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm TRM  --dataset PACS --test_envs ${test_env} --trial_seed 2 --cos_lam 1e-5 --dro_eta 1e-2
