from errno import EILSEQ
from multiprocessing.sharedctypes import Value
from webbrowser import get
import torch
import torchvision
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils import get_counts

def color_grayscale_arr(arr, color=0):
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if color == 0:
        arr = np.concatenate([arr,
                            np.zeros((h, w, 2), dtype=dtype)], axis=2)
    elif color == 1:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                            arr,
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)
    elif color == 2:
        arr = np.concatenate([arr,
                            arr,
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)
    elif color == 3:
        arr = np.concatenate([arr,
                            np.zeros((h, w, 1), dtype=dtype),
                            arr], axis=2)
    elif color == 4:
        arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype),
                            arr], axis=2)

    return arr

colors = {0: 'red', 1: 'green', 2: 'yellow', 3: 'pink', 4: 'blue'}

class MNIST:

    def __init__(self, root, cfg, split='train', transform=None, biased_val=False):
        random.seed(0)
        self.dataset = torchvision.datasets.MNIST(root, download=True, train=split=='train')
        self.random_idxs = random.sample(range(len(self.dataset)), int(len(self.dataset)*0.1))
        self.imgs = self.dataset.data[self.random_idxs]
        self.labels = self.dataset.targets[self.random_idxs]
        self.root = root
        self.split = split
        # if split == 'train':
        #     self.imgs = self.dataset.data
        #     self.labels = self.dataset.targets
        # elif split == 'val':
        #     self.imgs = self.dataset.data[:int(len(self.dataset.data)/2)]
        #     self.labels = self.dataset.targets[:int(len(self.dataset.data)/2)]
        # elif split == 'test':
        #     self.imgs = self.dataset.data[int(len(self.dataset.data)/2):]
        #     self.labels = self.dataset.targets[int(len(self.dataset.data)/2):]
        self.class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.transform = transform
        self.class_weights = get_counts(self.labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, label, old_idx = self.imgs[idx], self.labels[idx], self.random_idxs[idx]
        # filename = f"{self.root}/MNIST/{self.split}-{old_idx}.jpg"
        # img = Image.open(filename).convert('RGB')
        img = Image.fromarray(img.numpy(), mode="L").resize((224, 224))
        filename = f"{self.root}/MNIST/{self.split}-{old_idx}.jpg"
        if not os.path.exists(f"{self.root}/MNIST/"):
            os.makedir(f"{self.root}/MNIST/")
        if not os.path.exists(filename):
            img.save(filename)
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "domain": 0,
            "group": label, # since we dont have group labels for test set,
            "filename": filename
        }

class SVHN:

    def __init__(self, root, cfg, split='train', transform=None, biased_val=False):
        self.dataset = torchvision.datasets.SVHN(root, download=True, split='train' if split == 'train' else 'test')
        self.random_idxs = random.sample(range(len(self.dataset)), int(len(self.dataset)*0.1))
        self.root = root
        self.split = split
        self.imgs = self.dataset.data[self.random_idxs]
        self.labels = self.dataset.labels[self.random_idxs]
        # if split == 'train':
        #     self.imgs = self.dataset.data
        #     self.labels = self.dataset.labels
        # elif split == 'test':
        #     self.imgs = self.dataset.data[:int(len(self.dataset.data)/2)]
        #     self.labels = self.dataset.labels[:int(len(self.dataset.data)/2)]
        self.class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.transform = transform
        self.class_weights = get_counts(self.labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, label, old_idx = self.imgs[idx], self.labels[idx], self.random_idxs[idx]
        # filename = f"{self.root}/SVHN/{self.split}-{old_idx}.jpg"
        # img = Image.open(filename).convert('RGB')
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).resize((224, 224))
        filename = f"{self.root}/SVHN/{self.split}-{old_idx}.jpg"
        if not os.path.exists(filename):
            img.save(filename)
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "domain": 1,
            "group": label, # since we dont have group labels for test set
            "filename": filename,
        }

class ColoredMNISTSimplified:

    def __init__(self, root, cfg, split='train', transform=None, biased_val=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.cfg = cfg
        if cfg.DATA.CONFOUNDING == 1.0 and self.split == 'train':
            try:
                self.data = np.load(f'{root}/ColoredMNIST/{split}_{self.cfg.DATA.BIAS_TYPE}_100.npy', allow_pickle=True).item()
            except:
                self.data = np.load(f'{root}/ColoredMNIST/{split}_{self.cfg.DATA.BIAS_TYPE}_biased.npy', allow_pickle=True).item()
        elif cfg.DATA.CONFOUNDING == 0.95 and self.split == 'train':
            self.data = np.load(f'{root}/ColoredMNIST/{split}_{self.cfg.DATA.BIAS_TYPE}_95.npy', allow_pickle=True).item()
        else:
            if split == 'val' and biased_val:
                self.data = np.load(f'{root}/ColoredMNIST/{split}_{self.cfg.DATA.BIAS_TYPE}_biased.npy', allow_pickle=True).item()
            else:
                self.data = np.load(f'{root}/ColoredMNIST/{split}_{self.cfg.DATA.BIAS_TYPE}.npy', allow_pickle=True).item()
        self.imgs = self.data['images']
        self.labels = self.data['labels']
        self.domains = self.data['colors']
        self.domains = np.array([np.where(np.unique(self.domains) == d)[0][0] for d in self.domains])
        print(self.imgs.shape, self.labels.shape, self.domains.shape)
        if self.cfg.DATA.BIAS_TYPE == 'quinque':
            self.colors = {0: 'red', 1: 'green', 2: 'yellow', 3: 'pink', 4: 'blue'}
            self.groups = np.array([np.array([r*5 + i for i in range(5)]) for r in range(10)])
        elif self.cfg.DATA.BIAS_TYPE == 'binary':
            self.colors = {0: 'red', 1: 'green'}
            self.groups = np.array([np.array([r*2 + i for i in range(2)]) for r in range(10)])
        # else: raise ValueError(f"{self.cfg.DATA.BIAS_TYPE} not a valid bias type.")
        else:
            self.colors = {0: 'red', 1: 'blue'}
            self.groups = np.array([np.array([r*2 + i for i in range(2)]) for r in range(10)])
        self.class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if self.cfg.DATA.SHRINK:
            self.imgs, self.labels, self.domains = self.filter()
        self.class_weights = get_counts(self.labels)

    def filter(self):
        np.random.seed(0)
        filtered_im, filtered_label, filtered_dom = [], [], []
        for (im, label, dom) in zip(self.imgs, self.labels, self.domains):
            filtered_im.append(im)
            filtered_label.append(label)
            filtered_dom.append(dom)
        # take a sample so things run faster
        print(len(filtered_im), len(filtered_dom))
        sample_idxs = np.random.choice(np.array(list(range(len(filtered_im)))), int(0.5 * len(filtered_im)))
        filtered_im = np.array(filtered_im)[sample_idxs]
        filtered_dom = np.array(filtered_dom)[sample_idxs]
        filtered_label = np.array(filtered_label)[sample_idxs]
        print(f"{self.split} set len: {len(filtered_im)}")
        return filtered_im, filtered_label, filtered_dom
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx], 'RGB').resize((224, 224))
        filename = f"{self.root}/ColoredMNISTSimpleData/shrink-{str(self.cfg.DATA.SHRINK)}-{self.cfg.DATA.BIAS_TYPE}-{self.split}-{idx}.jpg"
        if not os.path.exists(filename):
            img.save(filename)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        domain = self.domains[idx]
        if (label % 2 == 0 and domain == 1) or (label % 2 == 1 and domain == 0):
            domain = 1
        else:
            domain = 0
        return {
            "image": img,
            "label": label,
            "group": domain,
            "domain": domain,
            "filename": filename,
        }

class ColoredMNIST(Dataset):

    def __init__(self, root, cfg, split='train', transform=None, biased_val=False):
        self.data_split = split
        data_dic = np.load(os.path.join(root,'mnist_10color_jitter_var_%.03f.npy'%cfg.DATA.COLOR_VAR),encoding='latin1', allow_pickle=True).item()
        if self.data_split == 'train':
            self.image = np.array([img for i, img in enumerate(data_dic['train_image']) if i % 6 != 0])
            self.label = np.array([lab for i, lab in enumerate(data_dic['train_label']) if i % 6 != 0])

        if self.data_split == 'test':
            # self.image = np.array([img for i, img in enumerate(data_dic['test_image']) if i % 2 == 0])
            # self.label = np.array([lab for i, lab in enumerate(data_dic['test_label']) if i % 2 == 0])
            self.image = np.array(data_dic['test_image'])
            self.label = np.array(data_dic['test_label'])

        if self.data_split =='val':
            if biased_val: # take from train set
                self.image = np.array([img for i, img in enumerate(data_dic['train_image']) if i % 6 == 0])
                self.label = np.array([lab for i, lab in enumerate(data_dic['train_label']) if i % 6 == 0])
            else: # take from test set
                self.image = np.array([img for i, img in enumerate(data_dic['test_image']) if i % 2 == 1])
                self.label = np.array([lab for i, lab in enumerate(data_dic['test_label']) if i % 2 == 1])  

        self.class_weights = get_counts(self.label)
        self.transform = transform
        color_var = cfg.DATA.COLOR_VAR
        self.color_std = color_var**0.5
        self.class_labels = [str(i) for i in range(10)]

        self.ToPIL = transforms.Compose([
                              transforms.ToPILImage(),
                              ])


    def __getitem__(self,index):
        label = self.label[index]
        image = self.image[index]

        image = self.ToPIL(image)

        label_image = image.resize((14,14), Image.NEAREST) 

        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)))
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image + mask_image
        label_image = label_image.long()

        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": label,
            "domain": label,
            "group": label # since we dont have group labels for test set
        }

    def __len__(self):
        return self.image.shape[0]