# Transfer Risk Minimization



## 10C-CMNIST & SceneCOCO

### Prepare the dataset

- Preprocess the COCO dataset :

```shell
python coco.py
```

- Preprocess the Places dataset:

```shell
python places.py
```

- Experiments:

```shell
python -m domainbed.scripts.train   --data_dir=root --algorithm alg  --dataset dataset --trial_seed t_seed --bias bias  --shift shift --epochs epochs

root: root directory for the data
dataset: ColoredMNIST or SceneCOCO
alg: ERM, VREx, IRM, GroupDRO, Fish, MLDG, TRM
bias: bias degree r
shift: 0: label-correlatd shift, 1: label-uncorrelated shift, 2: combined-shift
t_seed: seed for data splitting / feature combinations
epochs: training epochs
```



## PACS / Office-Home

```shell
python -m domainbed.scripts.train  --data_dir=root --algorithm alg  --dataset dataset --trial_seed t_seed (--resnet18) --epochs epochs

root: root directory for the data
alg: ERM, VREx, IRM, GroupDRO, Fish, MLDG, TRM
t_seed: seed for data splitting
dataset: PACS or OfficeHome
resnet18: use ResNet18 (default: ResNet50)
epochs: training epochs
```



## Group Distributional Robusness

```shell
python -m domainbed.scripts.train  --data_dir=../domainbed --algorithm alg --dataset Celeba --trial_seed t_seed --robust --resnet18 (--reweight) 
 
alg: TRM_DRO, ERM, GroupDRO
t_seed: seed for data splitting
reweight: reweight the objective
```



This implementation is based on / inspired by:

- https://github.com/facebookresearch/DomainBed (code structure).



