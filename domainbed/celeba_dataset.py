from functools import partial
import torch
import os
import PIL
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
import numpy as np
import pandas
# Domains
domain_fn = {
    # lambda t: (t[:, 20] == 1) & (t[:, 9] == 1),  # Male            Blond
    # lambda t: (t[:, 20] == 1) & (t[:, 9] == 0),  # Male            Not-Blond
    # lambda t: (t[:, 20] == 0) & (t[:, 9] == 1),  # Female          Blond
    # lambda t: (t[:, 20] == 0) & (t[:, 9] == 0),  # Female          Not-Blond
    # 'glass': lambda t: (t[:, 15] == 1),
    # 'nonglass': lambda t: (t[:, 15] == 0),
    #'male': lambda t: (t[:, 20] == 1),
    #'female': lambda t: (t[:, 20] == 0),
    # 'refine': lambda t: (t[:, 4] == 0) & (t[:, 35] == 0),
    # 'male_refine': lambda t: (t[:, 20] == 1) & (t[:, 4] == 0) & (t[:, 35] == 0),  # No Hats, No bald
    # 'female_refine': lambda t: (t[:, 20] == 0) & (t[:, 4] == 0) & (t[:, 35] == 0),
    # #'male_nonblond': lambda t: (t[:, 20] == 1) & (t[:, 9] == 0),
    # 'male_nonblond_refine': lambda t: (t[:, 20] == 1) & (t[:, 9] == 0) & (t[:, 4] == 0) & (t[:, 35] == 0),
    'male_blond': lambda t: (t[:, 20] == 1) & (t[:, 9] == 1),
    'male_nonblond': lambda t: (t[:, 20] == 1) & (t[:, 9] != 1),
    'female_blond': lambda t: (t[:, 20] == 0) & (t[:, 9] == 1),
    'female_nonblond': lambda t: (t[:, 20] == 0) & (t[:, 9] != 1),
    # 'female_nonblond_refine': lambda t: (t[:, 20] == 0) & (t[:, 9] == 0) & (t[:, 4] == 0) & (t[:, 35] == 0),
    # 'young': lambda t: (t[:, 39] == 1),
    # 'old': lambda t: (t[:, 39] == 0),
    # 'blond': lambda t: (t[:, 9] == 1),
    # 'nonblond': lambda t: (t[:, 9] == 0),
    # 'blond_refine': lambda t: (t[:, 9] == 1) & (t[:, 4] == 0) & (t[:, 35] == 0),
    # 'nonblond_refine': lambda t: (t[:, 9] == 0) & (t[:, 4] == 0) & (t[:, 35] == 0),
    # 'black': lambda t: (t[:, 8] == 1),
    # 'young_blond': lambda t: (t[:, 39] == 1) & (t[:, 9] == 1),  # Young           Blond
    #'young_nonblond': lambda t: (t[:, 39] == 1) & (t[:, 9] == 0),  # Young           Not-Blond
    # 'old_blond': lambda t: (t[:, 39] == 0) & (t[:, 9] == 1),  # Old             Blond
    # #'old_nonblond': lambda t: (t[:, 39] == 0) & (t[:, 9] == 0),  # Old             Not-Blond
    # 'young_black': lambda t: (t[:, 39] == 1) & (t[:, 8] == 1),  # Young           Black
    # 'old_black': lambda t: (t[:, 39] == 0) & (t[:, 8] == 1),  # Old             Black
    # 'glass_blond': lambda t: (t[:, 15] == 1) & (t[:, 9] == 1),
    # 'nonglass_nonblond': lambda t: (t[:, 15] == 0) & (t[:, 9] == 0),
    # lambda t: (t[:, 2] == 1) & (t[:, 9] == 1),  # Attractive      Blond
    # lambda t: (t[:, 2] == 1) & (t[:, 9] == 0),  # Attractive      Not-Blond
    # lambda t: (t[:, 2] == 0) & (t[:, 9] == 1),  # Not-Attractive  Blond
    # lambda t: (t[:, 2] == 0) & (t[:, 9] == 0),  # Not-Attractive  Not-Blond
    # lambda t: (t[:, 32] == 1) & (t[:, 9] == 1),  # Straight-Hair   Blond
    # lambda t: (t[:, 32] == 1) & (t[:, 9] == 0),  # Straight-Hair   Not-Blond
    # lambda t: (t[:, 33] == 1) & (t[:, 9] == 1),  # Wavy-Hair       Blond
    # lambda t: (t[:, 33] == 1) & (t[:, 9] == 0),  # Wavy-Hair       Not-Blond
}

class CelebA_process_group(VisionDataset):

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(CelebA_process_group, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')


        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

        self.create_files(domain_fn)


    def create_files(self, groups_function:dict):

        assert len(groups_function.keys())>0

        for k, v in groups_function.items():
            fn = partial(os.path.join, self.root, self.base_folder)
            mask = v(self.attr)
            file_names = self.filename[mask]
            attr = self.attr[mask]
            assert len(file_names) == len(attr)

            # save file paths
            dict_file = {'file_path': file_names}
            df = pandas.DataFrame(dict_file)
            df.to_csv(fn('file_path_'+k+'.csv'))

            # save attr
            attr = attr.numpy()
            np.save(fn('attr_'+k), attr)

            print(f"Group :{k}, files save to {fn('file_path_'+k+'.csv')}, length: {len(file_names)}")

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))


    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self):
        return len(self.attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class CelebA_group(VisionDataset):

    base_folder = "celeba"

    def __init__(self, root, group_name, transform=None, target_transform=None, label=0, null=False):

        super(CelebA_group, self).__init__(root, transform=transform,
                                     target_transform=target_transform)

        fn = partial(os.path.join, self.root, self.base_folder)

        self.filename = pandas.read_csv(fn('file_path_'+group_name+'.csv'), header=0)['file_path'].values
        self.attr = torch.from_numpy(np.load(fn('attr_'+group_name+'.npy'), allow_pickle=True))
        assert len(self.filename) == len(self.attr), f"filenames and attr is not aligned:{len(self.filename)}, {len(self.attr)}"
        if label == 0:
            self.label = torch.zeros(len(self.filename)).long()
        else:
            self.label = torch.ones(len(self.filename)).long()

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        label = self.label[index]

        if self.transform is not None:
            X = self.transform(X)

        return X, label

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


if __name__ == '__main__':

    # generate files/attr files according to the domain_fn
    CelebA_process_group('/home/ylxu/domainbed', split='all', target_type='attr',
                        transform=None, download=False)