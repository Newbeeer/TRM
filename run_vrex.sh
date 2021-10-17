#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset PlacesCOCO --test_envs 0 --trial_seed 1 --bias 0.5 --rex_lam 0.1

CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset PlacesCOCO --test_envs 0 --trial_seed 2 --bias 0.5 --rex_lam 0.1

CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset PlacesCOCO --test_envs 0 --trial_seed 3 --bias 0.5 --rex_lam 0.1

CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset PlacesCOCO --test_envs 0 --trial_seed 4 --bias 0.5 --rex_lam 0.1

CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm VREx  --dataset PlacesCOCO --test_envs 0 --trial_seed 5 --bias 0.5 --rex_lam 0.1




