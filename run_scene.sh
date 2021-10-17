#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train <gpu_num>"
  exit 1
fi

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 25 --bias 0.6 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset SceneCOCO --test_envs 0 --trial_seed 25 --bias 0.6 --irm_lam 0.1 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset SceneCOCO --test_envs 0 --trial_seed 25 --bias 0.6 --rex_lam 0.1 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset SceneCOCO --test_envs 0 --trial_seed 25 --bias 0.6 --dro_eta 1e-2 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM  --dataset SceneCOCO --test_envs 0 --trial_seed 25 --bias 0.6 --dro_eta 1e-2 --shift 2 --test_val


CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm ERM  --dataset SceneCOCO --test_envs 0 --trial_seed 26 --bias 0.6 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm IRM  --dataset SceneCOCO --test_envs 0 --trial_seed 26 --bias 0.6 --irm_lam 0.1 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset SceneCOCO --test_envs 0 --trial_seed 26 --bias 0.6 --rex_lam 0.1 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset SceneCOCO --test_envs 0 --trial_seed 26 --bias 0.6 --dro_eta 1e-2 --shift 2 --test_val

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm TRM  --dataset SceneCOCO --test_envs 0 --trial_seed 26 --bias 0.6 --dro_eta 1e-2 --shift 2 --test_val

