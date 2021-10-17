


#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset VLCS --test_envs ${test_env} --trial_seed 2 --resnet18
#
CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm IRM \
  --dataset VLCS --test_envs ${test_env} --trial_seed 3 --resnet18

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm MLDG \
  --dataset VLCS --test_envs ${test_env} --trial_seed 3 --resnet18

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm Fish \
  --dataset VLCS --test_envs ${test_env} --trial_seed 3 --resnet18

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm TRM \
  --dataset VLCS --test_envs ${test_env} --trial_seed 3 --resnet18 --class_balanced

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs ${test_env} --trial_seed 2 --resnet18
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs ${test_env} --trial_seed 5 --resnet18
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs ${test_env} --trial_seed 4 --resnet18


#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs 3 --trial_seed 5 --cos_lam 1e-5 --resnet18 --dro_eta 1e-2
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs 3 --trial_seed 5 --cos_lam 1e-5 --resnet18 --dro_eta 1e-2
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs 3 --trial_seed 4 --cos_lam 1e-5 --resnet18 --dro_eta 1e-2

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs 2 --trial_seed 2 --cos_lam 1e-5 --resnet18 --dro_eta 1e-2
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs 2 --trial_seed 5 --cos_lam 1e-5 --resnet18 --dro_eta 1e-2
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs 2 --trial_seed 5 --cos_lam 1e-5 --resnet18 --dro_eta 1e-2
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train   --data_dir=../data --algorithm ERM \
#  --dataset PACS --test_envs 2 --trial_seed 4 --cos_lam 1e-5 --resnet18 --dro_eta 1e-2

