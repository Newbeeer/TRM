
gpu=$1
test_env=$2
cos=$3


CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm TRM \
  --dataset OfficeHome --test_envs ${test_env} --trial_seed 2 --resnet18 \
  --class_balanced --cos_lam ${cos} --iters 1000

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm TRM \
#  --dataset OfficeHome --test_envs 1 --trial_seed 2 --resnet18 --class_balanced --cos_lam ${cos}

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm TRM \
#  --dataset OfficeHome --test_envs 2 --trial_seed 2 --resnet18 --class_balanced --cos_lam ${cos}
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm TRM \
#  --dataset OfficeHome --test_envs 3 --trial_seed 2 --resnet18 --class_balanced --cos_lam ${cos}

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm Fish \
#  --dataset PACS --test_envs ${test_env} --trial_seed 3 --resnet18 --fish_lam 0.1

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm TRM \
#  --dataset OfficeHome --test_envs ${test_env} --trial_seed 5 --resnet18 --class_balanced


