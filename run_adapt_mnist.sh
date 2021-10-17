#!/usr/bin/env bash


#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path VREx_bias_0.6_seed_2_SceneCOCO --n_data 100 --resnet18
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path GroupDRO_bias_0.6_seed_2_SceneCOCO --n_data 100 --resnet18
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm Swap  --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path Swap_bias_0.6_seed_2_SceneCOCO --n_data 100 --resnet18
#
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path ERM_bias_0.6_seed_3_SceneCOCO --n_data 100 --resnet18
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path IRM_bias_0.6_seed_3_SceneCOCO --n_data 100 --resnet18
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path VREx_bias_0.6_seed_3_SceneCOCO --n_data 100 --resnet18
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path GroupDRO_bias_0.6_seed_3_SceneCOCO --n_data 100 --resnet18

#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm MLDG \
# --dataset SceneCOCO --test_envs 0 --trial_seed 3 --bias 0.6  --shift 0  --model_save

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 3 --resume --resume_path MLDG_bias_0.6_seed_3_SceneCOCO --n_data 100

 CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path MLDG_bias_0.6_seed_3_SceneCOCO --n_data 100
 
CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 3 --resume --resume_path MLDG_bias_0.6_seed_3_SceneCOCO --n_data 300

 CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path MLDG_bias_0.6_seed_3_SceneCOCO --n_data 300

 CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 3 --resume --resume_path MLDG_bias_0.6_seed_3_SceneCOCO --n_data 500

 CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.retrain   --data_dir=../domainbed --algorithm ERM \
 --dataset SceneCOCO --test_envs 0 --trial_seed 1 --resume --resume_path MLDG_bias_0.6_seed_3_SceneCOCO --n_data 500

