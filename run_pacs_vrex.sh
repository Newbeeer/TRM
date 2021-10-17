CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 2 --trial_seed 3 --cos_lam 1e-5 --resnet18 --rex_lam 1.

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 2 --trial_seed 2 --cos_lam 1e-5 --resnet18 --rex_lam 1

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 2 --trial_seed 5 --cos_lam 1e-5 --resnet18 --rex_lam 1

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 2 --trial_seed 4 --cos_lam 1e-5 --resnet18 --rex_lam 1


CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 0 --trial_seed 3 --cos_lam 1e-5 --resnet18 --rex_lam 1

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 0 --trial_seed 2 --cos_lam 1e-5 --resnet18 --rex_lam 1

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 0 --trial_seed 5 --cos_lam 1e-5 --resnet18 --rex_lam 1

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
  --dataset PACS --test_envs 0 --trial_seed 4 --cos_lam 1e-5 --resnet18 --rex_lam 1

#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
#  --dataset PACS --test_envs 2 --trial_seed 3 --cos_lam 1e-5 --resnet18 --rex_lam 0.1

#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
#  --dataset PACS --test_envs 2 --trial_seed 2 --cos_lam 1e-5 --resnet18 --rex_lam 1
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
#  --dataset PACS --test_envs 2 --trial_seed 5 --cos_lam 1e-5 --resnet18 --rex_lam 1
#
#CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.train   --data_dir=../data --algorithm VREx \
#  --dataset PACS --test_envs 2 --trial_seed 4 --cos_lam 1e-5 --resnet18 --rex_lam 1

