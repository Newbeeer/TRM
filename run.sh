#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path ERM_bias_0.6_seed_2_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path IRM_bias_0.6_seed_2_PlacesCOCO --n_data 500 --resnet18

CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 12 --resume --resume_path VREx_bias_0.6_seed_2_PlacesCOCO --n_data 300 --resnet18

CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 12 --resume --resume_path VREx_bias_0.6_seed_3_PlacesCOCO --n_data 300 --resnet18
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path GroupDRO_bias_0.6_seed_2_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm Swap  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path Swap_bias_0.6_seed_2_PlacesCOCO --n_data 500 --resnet18
#
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path ERM_bias_0.6_seed_3_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path IRM_bias_0.6_seed_3_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path VREx_bias_0.6_seed_3_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path GroupDRO_bias_0.6_seed_3_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm Swap  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path Swap_bias_0.6_seed_3_PlacesCOCO --n_data 500 --resnet18
#
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path ERM_bias_0.6_seed_4_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path IRM_bias_0.6_seed_4_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path VREx_bias_0.6_seed_4_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path GroupDRO_bias_0.6_seed_4_PlacesCOCO --n_data 500 --resnet18
#
#CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm Swap  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --resume --resume_path Swap_bias_0.6_seed_4_PlacesCOCO --n_data 500 --resnet18