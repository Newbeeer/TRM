# Transfer Risk Minimization (TRM)



Code for [**Learning Representations that Support Robust Transfer of Predictors**](https://github.com/Newbeeer/TRM)



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

