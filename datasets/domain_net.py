import os
from PIL import Image
import numpy as np
from pytorch_adapt.datasets import  domainnet
from utils import get_counts

with open(f'/shared/lisabdunlap/data/domainnet/clipart_test.txt') as f:
    test_classes = f.read().splitlines()

class_dict = {}
for t in test_classes:
    class_dict[t.split(' ')[1]] = t.split(' ')[0].split('/')[-2]

DOMAINNET_CLASSES = list(class_dict.values())

class DomainNet:

    def __init__(self, root, cfg, split='train', transform=None):
        self.root = root
        self.cfg = cfg
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.data = domainnet.DomainNet(self.root, self.cfg.DATA.SOURCE_DOMAIN, True, transform=None)
        elif self.split == 'val':
            self.data = domainnet.DomainNet(self.root, self.cfg.DATA.SOURCE_DOMAIN, False, transform=None)
        elif self.split == 'test':
            self.data = domainnet.DomainNet(self.root, self.cfg.DATA.TARGET_DOMAIN, False, transform=None)
        self.labels = self.data.labels
        self.class_labels = DOMAINNET_CLASSES
        self.class_weights = get_counts(self.labels)
        self.img_paths = self.data.img_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "filename": self.img_paths[idx],
            "domain": 0,
            "group": label # since we dont have group labels for test set
        }

MINI_DOMAINS = ['clipart', 'painting', 'real', 'sketch'] 

with open(f'/shared/lisabdunlap/data/domainnet_sentry_split/real_test_mini.txt') as f:
    test_classes = f.read().splitlines()

class_dict = {}
for t in test_classes:
    class_dict[t.split(' ')[1]] = t.split(' ')[0].split('/')[-2]

MINI_DOMAINNET_CLASSES = sorted(list(class_dict.values()))

class OneDomain:

    def __init__(self, root: str, domain: str, split: str, transform):
        assert domain in ['clipart', 'painting', 'real', 'sketch'], 'domain must be one of clipart, real, painting, sketch'
        name = "train" if split == 'train' else "test"
        labels_file = os.path.join(root, "domainnet_sentry_split", f"{domain}_{name}_mini.txt")
        img_dir = os.path.join(root, "domainnet")
        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        self.label_class_names = [p.split('/')[-2] for p in self.img_paths]
        self.labels = [MINI_DOMAINNET_CLASSES.index(n) for n in self.label_class_names]
        self.transform = transform
        self.class_weights = get_counts(self.labels)
        self.class_labels = MINI_DOMAINNET_CLASSES
        self.domain = MINI_DOMAINS.index(domain)
        self.domains = [self.domain for i in range(len(self.img_paths))]
        print(f'... loading domain {domain} of size {len(self.img_paths)}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        label = self.labels[idx]
        domain = self.domains[idx]

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "domain": domain,
            "group": domain, # since we dont have group labels for test set
            "filename": self.img_paths[idx]
        }

class MultiDomain:

    def __init__(self, root: str, domains: list, split: str, transform):
        assert all([d in ['clipart', 'painting', 'real', 'sketch'] for d in domains]), 'domains must be one of clipart, real, painting, sketch'
        self.img_paths, self.domains, self.labels = [], [], []
        for d in domains:
            dataset = OneDomain(root, d, split, transform)
            self.img_paths.extend(dataset.img_paths)
            self.domains.extend([MINI_DOMAINS.index(d)] * len(dataset))
            self.labels.extend(dataset.labels)
        self.class_weights = get_counts(self.labels)
        self.transform = transform
        self.class_labels = MINI_DOMAINNET_CLASSES
        self.domain_labels = MINI_DOMAINS

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        label = self.labels[idx]
        domain = self.domains[idx]

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "domain": domain,
            "group": domain, # since we dont have group labels for test set
            "filename": self.img_paths[idx]
        }

class DomainNetMiniAug(OneDomain):
    """
    DomainNetMini training set augmented with StyleGAN-NADA combined with sketch train set.
    Since we dont have the labels for the aug dataset, we match the filename with the sketch
    dataset to get the label.
    """
    def __init__(self, root: str, domain: str, split: str, transform):
        self.root = root
        self.transform = transform
        self.aug_img_paths, self.aug_labels, self.aug_domains = [], [], []
        filenames_sketch = []
        for dom in ['clipart', 'painting', 'real']:
            filenames_sketch += [f.replace(dom, '').replace('.png', '.jpg') for f in os.listdir(os.path.join(self.root, dom))]
            filenames = [f for f in os.listdir(os.path.join(self.root, dom))]
            self.aug_img_paths += [f"{self.root}/{dom}/{f}" for f in filenames]
            # self.aug_labels += [int(f.split('_')[1]) for f in filenames]
            self.aug_domains += [MINI_DOMAINS.index(dom) for f in filenames]

        # hacky way of joining the two datasets since i count figure out torch.utils.data.ConcatDataset
        super().__init__(root='/shared/lisabdunlap/data', domain='sketch', split='train', transform=transform)
        # matches filenames to get labels
        img_file_paths = [f.split('/')[-1] for f in self.img_paths]
        remove_idxs, matching_idxs = [], []
        for idx, sketch_file in enumerate(filenames_sketch):
            try:
                matching_idx = img_file_paths.index(sketch_file)
                matching_idxs.append(matching_idx)
                self.aug_labels.append(self.labels[matching_idx])
            except:
                remove_idxs.append(idx)

        self.aug_img_paths = np.array([a for i, a in enumerate(self.aug_img_paths) if i not in remove_idxs])
        self.aug_domains = np.array([d for i, d in enumerate(self.aug_domains) if i not in remove_idxs])
        self.aug_labels = np.array(self.aug_labels)
        assert len(self.aug_labels) == len(self.aug_img_paths), f"lengths of aug labels ({len(self.aug_labels)}) and paths ({len(self.aug_img_paths)}) dont match"
        for domain in ['clipart', 'painting', 'real']:
            print(f'... loading domain {domain} with {len(self.aug_img_paths[np.where(self.aug_domains == MINI_DOMAINS.index(domain))])} augmented images')
        self.img_paths = np.concatenate([self.aug_img_paths, self.img_paths])
        self.labels = np.concatenate([self.aug_labels, self.labels])
        self.domains = np.concatenate([self.aug_domains, self.domains])

    # def __len__(self):
    #     return len(self.img_paths)

    # def __getitem__(self, idx):
    #     img = Image.open(self.img_paths[idx])
    #     label = self.labels[idx]
    #     domain = self.domains[idx]

    #     if self.transform:
    #         img = self.transform(img)

    #     return {
    #         "image": img,
    #         "label": label,
    #         "domain": domain,
    #         "group": domain, # since we dont have group labels for test set
    #         "filename": self.img_paths[idx]
    #     }