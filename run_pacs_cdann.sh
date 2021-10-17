test_env=$1
gpu=$2

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm MMD \
 --dataset PACS --test_envs ${test_env} --trial_seed 3 --dro_eta 1e-2 --resnet18

 CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm MMD \
 --dataset PACS --test_envs ${test_env} --trial_seed 2 --dro_eta 1e-2 --resnet18

# CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm TRM \
# --dataset PACS --test_envs ${test_env} --trial_seed 5 --dro_eta 1e-2 --resnet18
#
# CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm TRM \
# --dataset PACS --test_envs ${test_env} --trial_seed 4 --dro_eta 1e-2 --resnet18

