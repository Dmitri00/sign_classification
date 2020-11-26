import numpy as np
from abc import abstractmethod, ABC
import os
import torch
import torch.nn as nn
from PIL import Image
from collections import Counter
from torchvision import datasets, models, transforms
class ImgClassificationDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __init__(self):
        pass
class ImgClassificationSourceDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root, image_set, transforms):
        super().__init__()
        self.img_root, self.gt_name = self.get_ds_path(root, image_set)
        self.transforms = transforms
        self.imgs = self.list_imgs()
        self.ground_truth = self.parse_ground_truth(self.gt_name)

        self.setup_ds_metadata()

    def setup_ds_metadata(self):
        self.label_counter = Counter(self.ground_truth)
        self.labels = sorted(self.label_counter.keys())
        self.num_classes = len(self.labels)

    @abstractmethod
    def get_ds_path(self, root, image_set):
        pass
    @abstractmethod
    def parse_ground_truth(self, gt_name):
        pass
    @abstractmethod
    def list_imgs(self):
        pass

    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_root, img_name)
        label = self.ground_truth[idx]
        img = Image.open(img_path).convert("RGB")

        # there is only one class
        label = torch.tensor(label, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class TrafficSign(ImgClassificationSourceDataset):
    def __init__(self, root, image_set, transforms):
        super().__init__(root, image_set, transforms)
        # load all image files, sorting them to
        # ensure that they are aligned

    def list_imgs(self):
        files_in_dir = os.listdir(self.img_root)
        return sorted(files_in_dir)

    def get_ds_path(self, root, image_set):
        if image_set == 'train':
            img_root = os.path.join(root, 'train')
            gt_name = os.path.join(root, "gt_train.csv")
        elif image_set == 'val' or image_set == 'test':
            img_root = os.path.join(root, 'test')
            gt_name = os.path.join(root, "gt_test.csv")
        else:
            raise ValueError("Can't parse image set name: {:}".format(image_set))
        return img_root, gt_name

    def parse_ground_truth(self, gt_name):
        with open(gt_name, 'r') as gt_file:
            gt_lines = gt_file.readlines()
        gt_lines = gt_lines[1:]
        ground_truth = []
        for entry_line in gt_lines:
            entry_splited = entry_line.split(',')
            img_name = entry_splited[0]
            label = int(entry_splited[1])
            ground_truth.append(label)
        return ground_truth

    def subset(self, indices):
        return TrafficSignSubset(self, indices)

    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_root, img_name)
        label = self.ground_truth[idx]
        img = Image.open(img_path).convert("RGB")

        # there is only one class
        label = torch.tensor(label, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def subset_by_predicat(self, predicat_fn):
        new_indices = []
        for idx, img_name in enumerate(self.imgs):
            img_path = os.path.join(self.img_root, img_name)
            label = self.ground_truth[idx]
            img = Image.open(img_path).convert("RGB")
            if predicat_fn(img):
                new_indices.append(idx)
        return TrafficSignSubset(self, new_indices)

    def __len__(self):
        return len(self.imgs)


class TrafficSignSubset(torch.utils.data.Subset):
    def __init__(self, trafficsign_dataset, subset_indices):
        super().__init__(trafficsign_dataset, subset_indices)
        self.ground_truth = [trafficsign_dataset.ground_truth[idx] for idx in subset_indices]
        self.label_counter = Counter(self.ground_truth)
        self.labels = sorted(self.label_counter.keys())

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        label = self.labels.index(label)
        return img, label

    def __getattr__(self, name):
        return getattr(self.dataset, name)
