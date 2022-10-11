import os
import clip
import torch
import torchvision

import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import random
from mpl_toolkits.axes_grid1 import ImageGrid
import os

CUB_DOMAINS = ["photo", "painting"]

with open('/shared/lisabdunlap/data/CUB-200-Painting/classes.txt') as f:
    lines = f.readlines()
    
CUB_CLASSES = [l.replace('\n', '').split('.')[-1].replace('_', ' ') for l in lines]

class Cub2011(torch.utils.data.Dataset):
    base_folder = 'CUB_200_2011/images'

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img =  Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        # return img, target
        return {
            "image": img,
            "label": target,
            "group": 0,
            "domain": 0,
            "filename": path,
        }

class Cub2011Painting(torchvision.datasets.ImageFolder):

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        img =  Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "label": target,
            "group": 1,
            "domain": 1,
            "filename": path,
        }