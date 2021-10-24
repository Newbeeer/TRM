# Transfer Risk Minimization (TRM)



Code for [**Learning Representations that Support Robust Transfer of Predictors**](https://arxiv.org/abs/2110.09940), Yilun Xu, Tommi Jaakkola



## Prepare the Datasets



#### Preprocess the SceneCOCO dataset :

```shell
# preprocess COCO
python coco.py
# preprocess Places
python places.py

# generate SceceCOCO dataset
python cocoplaces.py
```



## Running the Experiments

- Datasets:
  - Synthetic datasets for controlled experiments: ColorMNIST / SceneCOCO

  <img src="https://github.com/Newbeeer/TRM/blob/main/img/correlated_row.png" width="650px" />

  - Real-world datasets: PACS / Office-Home

```shell
python -m domainbed.scripts.train  --data_dir {root} --algorithm {alg} \
	--dataset {dataset} --trial_seed {t_seed} --epochs {epochs}  (--resnet50)

root: root directory for the data
alg: ERM, VREx, IRM, GroupDRO, Fish, MLDG, TRM
t_seed: seed for data splitting
dataset: PACS or OfficeHome or ColoredMNIST or SceneCOCO
resnet50: use ResNet50 (default: ResNet18)
epochs: training epochs
```





This implementation is based on / inspired by:

- https://github.com/facebookresearch/DomainBed (code structure).

- https://github.com/Faruk-Ahmed/predictive_group_invariance (data generation)

