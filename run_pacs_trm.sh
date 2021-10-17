gpu=$1
test_env=$2

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm TRM \
 --dataset PACS --test_envs ${test_env} --trial_seed 3 --dro_eta 1e-2 --resnet18 --test_val

 CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm TRM \
 --dataset PACS --test_envs ${test_env} --trial_seed 2 --dro_eta 1e-2 --resnet18 --test_val

 CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm TRM \
 --dataset PACS --test_envs ${test_env} --trial_seed 3 --dro_eta 1e-2

 CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm TRM \
 --dataset PACS --test_envs ${test_env} --trial_seed 2 --dro_eta 1e-2

