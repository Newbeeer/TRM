#!/usr/bin/env bash


gpu=$1
cos=$2

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train_dro   --data_dir=../data --algorithm ERM \
#  --dataset Celeba --trial_seed 1 --img_size 224 --resnet18 --dro_eta ${cos}

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train_dro   --data_dir=../data --algorithm ERM \
#  --dataset Celeba --trial_seed 2 --img_size 224 --resnet18 --dro_eta ${cos}

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train_dro   --data_dir=../data --algorithm ERM \
  --dataset Celeba --trial_seed 3 --img_size 224 --resnet18 --dro_eta ${cos}

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train_dro   --data_dir=../data --algorithm TRM_DRO \
#  --dataset Celeba --trial_seed 3 --reweight --img_size 224 --resnet18 --cos_lam ${cos}
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train_dro   --data_dir=../data --algorithm TRM_DRO \
#  --dataset Celeba --trial_seed 1 --reweight --img_size 224 --resnet18 --cos_lam ${cos}
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train_dro   --data_dir=../data --algorithm TRM_DRO \
#  --dataset Celeba --trial_seed 0 --reweight --img_size 224 --resnet18 --cos_lam ${cos}
