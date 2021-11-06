# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict

import PIL
import torch
from PIL import Image, ImageFile
from domainbed.utils import TensorDataset, save_image
from torchvision import transforms
import torchvision.datasets.folder
from torchvision.datasets import CIFAR100, MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import tqdm
import io
import functools

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "RotatedMNIST",
    "ColoredMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "ColoredCOCO",
    "ScencCOCO"
]

NUM_ENVIRONMENTS = {
    # Debug
    "Debug28": 3,
    "Debug224": 3,
    # Small images
    "RotatedMNIST": 6,
    "ColoredMNIST": 3,
    # Big images
    "VLCS": 4,
    "PACS": 4,
    "OfficeHome": 4,
    "TerraIncognita": 4,
    "DomainNet": 6,
}


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class MultipleDomainDataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8


class Debug(MultipleDomainDataset):
    DATASET_SIZE = 16
    INPUT_SHAPE = None  # Subclasses should override

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.environments = [0, 1, 2]
        self.datasets = []
        for _ in range(len(self.environments)):
            self.datasets.append(
                TensorDataset(
                    torch.randn(self.DATASET_SIZE, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (self.DATASET_SIZE,))
                )
            )

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENT_NAMES = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENT_NAMES = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        self.colors = torch.FloatTensor(
            [[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0], [0, 255, 0], [65, 105, 225], [0, 225, 225],
             [0, 0, 255], [255, 20, 147], [180, 180, 180]])
        self.random_colors = torch.randint(255, (10, 3)).float()

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.environments = environments

        for i in range(len(self.environments)):
            images = original_images[i::len(self.environments)]
            labels = original_labels[i::len(self.environments)]
            #             images = original_images
            #             labels = original_labels
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class ColoredMNIST(MultipleEnvironmentMNIST):

    def __init__(self, root, bias, test_envs, hparams):
        # config setting:
        # 0: random seed for environmental color;
        # 1: use default colors (True) or random colors;
        # 2: Bernoulli parameters for environmental color;
        # 3: designated environmental color number;
        # 4: random seed for bkgd colors
        # 5: Color digit?
        # 6: Color bkgd?
        # 7: Bernoulli parameters for bkgd colors
        #
        seed = int(torch.randint(100, [1])[0])
        if hparams['shift'] == 0:
            config = [[seed, True, 0, None, seed, True, False, 0],
                          [seed, True, 1, None, seed, True, False, 0],
                          [seed, True, bias, None, seed, True, False, 0]]
        if hparams['shift'] == 1:
            config = [[seed, True, 0, None, int(torch.randint(100, [1])[0]), False, True, 0],
                          [seed, True, 0, None, int(torch.randint(100, [1])[0]), False, True, 0],
                          [seed, True, 0, None, int(torch.randint(100, [1])[0]), False, True, 0]]
        if hparams['shift'] == 2:
            config = [[seed, True, 0, None, int(torch.randint(100, [1])[0]), True, True, 0],
                          [seed, True, 1, None, int(torch.randint(100, [1])[0]), True, True, 0],
                          [seed, True, bias, None, int(torch.randint(100, [1])[0]), True, True, 0]]

        print("config:", config)
        self.vis = False
        self.input_shape = (3, 28, 28,)
        self.num_classes = 10
        super(ColoredMNIST, self).__init__(root, config, self.color_dataset, (3, 28, 28,), 10)

        # TODO: set up verbose mode

    def color_dataset(self, images, labels, environment):
        # set the seed

        original_seed = torch.cuda.initial_seed()
        torch.manual_seed(environment[0])
        shuffle = torch.randperm(len(self.colors))
        self.colors_ = self.colors[shuffle] if environment[1] else torch.randint(255, (10, 3)).float()
        torch.manual_seed(environment[0])
        ber_digit = self.torch_bernoulli_(environment[2], len(labels))

        torch.manual_seed(environment[4])
        shuffle = torch.randperm(len(self.colors))
        bkgd_colors = torch.randint(255, (10, 3)).float() * 0.75
        torch.manual_seed(environment[4])
        ber_bkgd = self.torch_bernoulli_(environment[7], len(labels))
        images = torch.stack([images, images, images], dim=1)

        torch.manual_seed(original_seed)
        # binarize the images
        images = (images > 0).float()
        masks = (1 - images)
        total_len = len(images)
        if self.vis:
            image_collect = torch.empty(20, *self.input_shape)
            current_label = 0
            current_cnt = 0
            total_cnt = 0
        # Apply the color to the image
        for img_idx in range(total_len):
            # change digit colors
            if ber_digit[img_idx] > 0:
                if environment[5]:
                    images[img_idx] = images[img_idx] * self.colors_[labels[img_idx].long()].view(-1, 1, 1)
            else:
                color = torch.randint(10, [1])[0] if environment[3] is None else environment[3]
                if environment[5]:
                    images[img_idx] = images[img_idx] * self.colors_[color].view(-1, 1, 1)
            # change bkpg colors
            if ber_bkgd[img_idx] > 0:
                if environment[6]:
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + masks[img_idx] * bkgd_colors[
                        labels[img_idx].long()].view(-1, 1, 1)
            else:
                color = torch.randint(5, [1])[0]
                if environment[6]:
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + masks[img_idx] * bkgd_colors[color].view(
                        -1, 1, 1)
            if self.vis:
                if labels[img_idx] != current_label:
                    continue
                # visualize 20 images for sanity check
                import matplotlib.pyplot as plt
                image_collect[total_cnt] = images[img_idx] / 255.
                current_cnt += 1
                total_cnt += 1
                if current_cnt == 2:
                    current_cnt = 0
                    current_label += 1
                if total_cnt == 20:
                    break
        if self.vis:
            save_image(image_collect,'colormnist_uncorrelated_train.png',nrow=10)
            print(f"Visualization for {environment} Done")
            exit(0)
        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(True, x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENT_NAMES = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=PIL.Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        self.environments = [f.name for f in os.scandir(root) if f.is_dir()]
        self.environments = sorted(self.environments)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        print("enviroments:", self.environments)
        for i, environment in enumerate(self.environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)


        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
        print("number of classes:", self.num_classes)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENT_NAMES = ["C", "L", "S", "V"]

    def __init__(self, root,bias, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENT_NAMES = ["A", "C", "P", "S"]

    def __init__(self, root, bias, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENT_NAMES = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENT_NAMES = ["A", "C", "P", "R"]

    def __init__(self, root, bias, test_envs, hparams):
        self.dir = os.path.join(root, "officehome/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENT_NAMES = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


import h5py
import numpy as np


class MultipleEnvironmentCOCO(MultipleDomainDataset):
    def __init__(self, environments, dataset_transform, input_shape,
                 num_classes, places=False):
        super().__init__()
        self.colors = torch.FloatTensor(
            [[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0], [0, 255, 0], [65, 105, 225], [0, 225, 225],
             [0, 0, 255], [255, 20, 147], [180, 180, 180]])
        self.random_colors = torch.randint(255, (10, 3)).float()
        h5pyfname = '../data/coco'
        train_file = os.path.join(h5pyfname, 'train.h5py')
        val_file = os.path.join(h5pyfname, 'validtest.h5py')
        test_file = os.path.join(h5pyfname, 'idtest.h5py')
        train_data = h5py.File(train_file, 'r')
        val_data = h5py.File(val_file, 'r')
        test_data = h5py.File(test_file, 'r')
        original_images = np.concatenate(
            (train_data['resized_images'].value, test_data['resized_images'].value, val_data['resized_images'].value),
            axis=0)
        original_labels = np.concatenate((train_data['y'].value, test_data['y'].value, val_data['y'].value), axis=0)

        original_masks = np.concatenate(
            (train_data['resized_mask'].value, test_data['resized_mask'].value, val_data['resized_mask'].value), axis=0)

        print('image size:{}, label size:{}, mask:{}'.format(original_images.shape, original_labels.shape,
                                                             original_masks.shape))

        if places:
            places_file = os.path.join('../data/places/cocoplaces', 'places.h5py')
            places_data = h5py.File(places_file, 'r')
            self.places = places_data['resized_place'].value
            print('place size:{}'.format(self.places.shape))
            self.places = torch.from_numpy(self.places)

        original_images = torch.from_numpy(original_images)
        original_labels = torch.from_numpy(original_labels)
        original_masks = torch.from_numpy(original_masks)
        shuffle = torch.randperm(len(original_images))

        total_len = len(original_images)

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        original_masks = original_masks[shuffle]
        self.datasets = []
        self.environments = environments

        for i in range(len(self.environments)):
            images = original_images[i::len(self.environments)]
            labels = original_labels[i::len(self.environments)]
            masks = original_masks[i::len(self.environments)]
            #             images = original_images
            #             labels = original_labels
            #   masks = original_masks
            self.datasets.append(dataset_transform(images, labels, masks, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class SceneCOCO(MultipleEnvironmentCOCO):
    def __init__(self, root, bias, test_envs, hparams):
        # config setting:
        # 0: random seed for shuffle places;
        # 1: use default places or random places (deprecated in ScencCOCO);
        # 2: Bernoulli parameters for places;
        # 3: designated places number,
        # 4: random seed for digit colors
        # 5: using places? 6: Color digit?
        # 7: Bernoulli parameters for bkgd colors
        # 8: removing data-augmentation or not

        seed = int(torch.randint(100, [1])[0])
        if hparams['shift'] == 0:
            config = [[seed, True, 0, None, seed, True, False, 0, True],
                          [seed, True, 0.9, None, seed, True, False, 0, False],
                          [seed, True, bias, None, seed, True, False, 0, False]]
        if hparams['shift'] == 1:
            config = [[seed, True, 0, None, int(torch.randint(100, [1])[0]), False, True, 0, True],
                          [seed, True, 0, None, int(torch.randint(100, [1])[0]), False, True, 0, False],
                          [seed, True, 0, None, int(torch.randint(100, [1])[0]), False, True, 0, False]]
        if hparams['shift'] == 2:
            config = [[seed, True, 0, None, int(torch.randint(100, [1])[0]), True, True, 0, True],
                          [seed, True, 0.9, None, int(torch.randint(100, [1])[0]), True, True, 0, False],
                          [seed, True, bias, None, int(torch.randint(100, [1])[0]), True, True, 0, False]]
        print("config:", config)
        self.vis = False
        self.input_shape = (3, 64, 64,)
        self.num_classes = 10
        super(SceneCOCO, self).__init__(config, self.color_dataset, (3, 64, 64,), 10, True)

    def color_dataset(self, images, labels, masks, environment):

        # shuffle the colors
        original_seed = torch.cuda.initial_seed()
        torch.manual_seed(environment[0])
        # inter-class shuffleing
        for i in range(len(self.places)):
            shuffle = torch.randperm(len(self.places[i]))
            self.places[i] = self.places[i][shuffle]
        places = self.places[:10]
        # set the bernoulli r.v.
        torch.manual_seed(environment[0])
        ber = self.torch_bernoulli_(environment[2], len(labels))
        print("bernoulli:", len(ber), sum(ber))

        torch.manual_seed(environment[4])
        shuffle = torch.randperm(len(self.colors))
        obj_colors = self.colors[shuffle]
        # obj_colors = torch.randint(255, (10, 3)).float()
        torch.manual_seed(environment[4])
        ber_obj = self.torch_bernoulli_(environment[7], len(labels))

        torch.manual_seed(original_seed)
        total_len = len(images)
        label_counter = [0 for i in range(10)]
        if self.vis:
            image_collect = torch.empty(20, *self.input_shape)
            current_label = 0
            current_cnt = 0
            total_cnt = 0

        # Apply the color to the image
        for img_idx in range(total_len):
            if ber[img_idx] > 0:
                if environment[5]:
                    label = labels[img_idx]
                    place_img = places[label, int(label_counter[label])] * 0.75
                    images[img_idx] = place_img * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx]
                    label_counter[label] += 1
            else:
                if environment[5]:
                    label = torch.randint(10, [1])[0] if environment[3] is None else environment[3]
                    place_img = places[label, int(label_counter[label])] * 0.75
                    images[img_idx] = place_img * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx]
                    label_counter[label] += 1
            if ber_obj[img_idx] > 0:
                if environment[6]:
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx] * obj_colors[labels[img_idx].long()].view(-1, 1, 1) / 255.0
            else:
                if environment[6]:
                    color = torch.randint(5, [1])[0]
                    images[img_idx] = images[img_idx] * (1 - masks[img_idx]) + images[img_idx] * masks[img_idx] * obj_colors[color].view(-1, 1, 1) / 255.0
            if self.vis:
                if labels[img_idx] != current_label:
                    continue
                # visualize 20 images for sanity check
                import matplotlib.pyplot as plt
                image_collect[total_cnt] = images[img_idx]
                current_cnt += 1
                total_cnt += 1
                if current_cnt == 2:
                    current_cnt = 0
                    current_label += 1
                if total_cnt == 20:
                    break
        if self.vis:
            save_image(image_collect, 'scenecoco_combined_test.png', nrow=10)
            print(f"Visualization for {environment} Done")
            exit(0)
        print("label cnt:", label_counter)
        x = images.float()
        y = labels.view(-1).long()

        return TensorDataset(environment[8], x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
