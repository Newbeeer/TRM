gpu=$1


#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm ERM \
#	--dataset PACS --trial_seed 3 --epochs 50
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm TRM \
#	--dataset PACS --trial_seed 3 --epochs 50
#
#CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm Fish \
#	--dataset PACS --trial_seed 3 --epochs 50

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm ERM \
	--dataset ColoredMNIST --trial_seed 3

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm TRM \
	--dataset ColoredMNIST --trial_seed 3

CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.train  --data_dir ../data --algorithm Fish \
	--dataset ColoredMNIST --trial_seed 3