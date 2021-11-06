gpu=$1


CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm ERM \
	--dataset PACS --trial_seed 2 --epochs 50

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm TRM \
	--dataset PACS --trial_seed 2 --epochs 50

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm Fish \
	--dataset PACS --trial_seed 2 --epochs 50