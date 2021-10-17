#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset PlacesCOCO --test_envs 0 --trial_seed 2 --bias 0.5 --dro_eta 1e-1

CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset PlacesCOCO --test_envs 0 --trial_seed 3 --bias 0.5 --dro_eta 1e-1

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset PlacesCOCO --test_envs 0 --trial_seed 4 --bias 0.9
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset PlacesCOCO --test_envs 0 --trial_seed 5 --bias 0.9
#
#
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../domainbed --algorithm GroupDRO  --dataset PlacesCOCO --test_envs 0 --trial_seed 2 --bias 0.8