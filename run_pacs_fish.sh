gpu=$1
envs=$2


#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm Fish \
# --dataset PACS --test_envs ${envs} --trial_seed 3 --cos_lam 1e-5 --dro_eta 1e-2 --resnet18 --test_val

 CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm Fish \
 --dataset PACS --test_envs ${envs} --trial_seed 2 --cos_lam 1e-5 --dro_eta 1e-2  --resnet18 --test_val

#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm Fish \
# --dataset PACS --test_envs ${envs} --trial_seed 3 --cos_lam 1e-5 --dro_eta 1e-2
#
# CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir=../data --algorithm Fish \
# --dataset PACS --test_envs ${envs} --trial_seed 2 --cos_lam 1e-5 --dro_eta 1e-2

# CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train  --data_dir=../data --algorithm Fish \
# --dataset PACS --test_envs 3 --trial_seed 5 --cos_lam 1e-5 --dro_eta 1e-2 --resnet18
#
# CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train  --data_dir=../data --algorithm Fish \
# --dataset PACS --test_envs 3 --trial_seed 4 --cos_lam 1e-5 --dro_eta 1e-2 --resnet18