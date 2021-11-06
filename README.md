# Transfer Risk Minimization (TRM)



Code for [**Learning Representations that Support Robust Transfer of Predictors**](https://arxiv.org/abs/2110.09940)

[Yilun Xu](http://yilun-xu.com), [Tommi Jaakkola](http://people.csail.mit.edu/tommi/tommi.html)

**TL,DR:**  We introduce a simple robust estimation criterion -- transfer risk -- that is specifically geared towards optimizing transfer to new environments. Effectively, the criterion amounts to finding a representation that minimizes the risk of applying any optimal predictor trained on one environment to another. The transfer risk essentially decomposes into two terms, a direct transfer term and a weighted gradient-matching term arising from the optimality of per-environment predictors. 



## Prepare the Datasets

#### Download PACS/Office-Home/MNIST dataset:

```shell
python scripts/download.py --data_dir {data_dir}
```



Places dataset can be downloaded at:

http://data.csail.mit.edu/places/places365/train_256_places365standard.tar ; 

COCO dataset can be downloaded at:

http://images.cocodataset.org/annotations/annotations_trainval2017.zip

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
	--dataset {dataset} --trial_seed {t_seed} --epochs {epochs}  (--resnet50) (--test_eval)

root: root directory for the data
alg: ERM, VREx, IRM, GroupDRO, Fish, MLDG, TRM
t_seed: seed for data splitting
dataset: PACS or OfficeHome or ColoredMNIST or SceneCOCO
epochs: training epochs
resnet50: set ResNet50 as the backbone (default: ResNet18)
test_eval: test-domain validation set (default: train-domain validation set)
```





This implementation is based on / inspired by:

- https://github.com/facebookresearch/DomainBed (code structure).

- https://github.com/Faruk-Ahmed/predictive_group_invariance (data generation)

