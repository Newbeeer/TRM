test_env=$1
gpu=$2
irm=$3

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
  --dataset PACS --test_envs ${test_env} --trial_seed 3 --cos_lam 1e-5 --resnet18 --irm_lam ${irm}

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs ${test_env} --trial_seed 2 --cos_lam 1e-5 --resnet18 --irm_lam ${irm}
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs ${test_env} --trial_seed 5 --cos_lam 1e-5 --resnet18 --irm_lam ${irm}
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs ${test_env} --trial_seed 4 --cos_lam 1e-5 --resnet18 --irm_lam ${irm}

#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 1 --trial_seed 3 --cos_lam 1e-5 --resnet18 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 1 --trial_seed 2 --cos_lam 1e-5 --resnet18 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 1 --trial_seed 5 --cos_lam 1e-5 --resnet18 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 1 --trial_seed 4 --cos_lam 1e-5 --resnet18 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 2 --trial_seed 3 --cos_lam 1e-5 --resnet18 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 2 --trial_seed 2 --cos_lam 1e-5 --resnet18 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 2 --trial_seed 5 --cos_lam 1e-5 --resnet18 --irm_lam 1
#
#CUDA_VISIBLE_DEVICES=1 python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
#  --dataset PACS --test_envs 2 --trial_seed 4 --cos_lam 1e-5 --resnet18 --irm_lam 1

