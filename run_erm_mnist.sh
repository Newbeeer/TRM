#!/usr/bin/env bash


#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 3 --bias 0.85
CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 3 --bias 0.9

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 3 --bias 0.9 --irm_lam 1

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset ColoredMNIST --test_envs 0 --trial_seed 3 --bias 0.9 --rex_lam 10

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset ColoredMNIST --test_envs 0 --trial_seed 3 --bias 0.9 --dro_eta 1e-2

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm Swap  --dataset ColoredMNIST --test_envs 0 --trial_seed 3 --bias 0.9 --cos_lam 1e-4

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.9

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.9 --irm_lam 1

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.9 --rex_lam 10

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.9 --dro_eta 1e-2

CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm Swap  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.9 --cos_lam 1e-4

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.85
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.85 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.85 --rex_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.85 --dro_eta 1e-2
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm Swap  --dataset ColoredMNIST --test_envs 0 --trial_seed 2 --bias 0.85 --cos_lam 1e-4
#
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 5 --bias 0.85
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 5 --bias 0.85 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset ColoredMNIST --test_envs 0 --trial_seed 5 --bias 0.85 --rex_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset ColoredMNIST --test_envs 0 --trial_seed 5 --bias 0.85 --dro_eta 1e-2
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm Swap  --dataset ColoredMNIST --test_envs 0 --trial_seed 5 --bias 0.85 --cos_lam 1e-4
#
