#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train <gpu_num>"
  exit 1
fi

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.85 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.85 --irm_lam 0.1 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.85 --rex_lam 10 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.85 --dro_eta 1e-2 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.85 --dro_eta 1e-2 --shift 0 --test_val


CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.9 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.9 --irm_lam 0.1 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.9 --rex_lam 10 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.9 --dro_eta 1e-2 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.9 --dro_eta 1e-2 --shift 0 --test_val


CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.95 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.95 --irm_lam 0.1 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.95 --rex_lam 10 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.95 --dro_eta 1e-2 --shift 0 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM  --dataset ColoredMNIST --test_envs 0 --trial_seed 11 --bias 0.95 --dro_eta 1e-2 --shift 0 --test_val