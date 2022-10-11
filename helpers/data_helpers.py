import os, io
from tqdm import tqdm
import torch as ch
import torchvision
from torchvision import transforms
import numpy as np
# # from robustness import datasets
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import clip

from datasets.waterbirds import Waterbirds, WaterbirdsBoring, WaterbirdsSimple, WaterbirdsOrig
from datasets.planes import Planes, PlanesOrig, PlanesGroundSeg
from datasets.colored_mnist import ColoredMNIST, ColoredMNISTSimplified, MNIST, SVHN
from datasets.cats_dogs import CatsDogs
from datasets.imagenet import Imagenet
from datasets.domain_net import DomainNet, DOMAINNET_CLASSES, MINI_DOMAINNET_CLASSES, MINI_DOMAINS, OneDomain, MultiDomain, DomainNetMiniAug
from datasets.meta_shift import MetaShiftAttribute
from datasets.cub import Cub2011Painting, Cub2011, CUB_DOMAINS, CUB_CLASSES

from helpers.text_templates import imagenet_classes
from helpers.data_paths import DATASET_PATHS, BIASED_DATASET_PATHS

def get_config(name="Waterbirds"):
    base_cfg  = OmegaConf.load('configs/base.yaml')
    if name == "Waterbirds":
        cfg       = OmegaConf.load('configs/waterbirds.yaml')
    elif name == "WaterbirdsTiny":
        cfg       = OmegaConf.load('configs/waterbirds_tiny.yaml')
    elif name == "Waterbirds95":
        cfg       = OmegaConf.load('configs/waterbirds_95.yaml')
    elif name == "PlanesExt":
         cfg       = OmegaConf.load('configs/planes_ext.yaml')
    elif "Planes" in name:
        cfg       = OmegaConf.load('configs/planes.yaml')
    elif "ColoredMNIST" in name:
        cfg       = OmegaConf.load('configs/colored_mnist.yaml')
    elif 'CatsDogs' in name:
        cfg       = OmegaConf.load('configs/cats_dogs.yaml')
    elif 'Imagenet' in name:
        cfg       = OmegaConf.load('configs/imagenet.yaml')
    elif name == 'DomainNet':
        cfg       = OmegaConf.load('configs/domainnet.yaml')
    else:
        raise ValueError(f"{name} Dataset config not found")
    args      = OmegaConf.merge(base_cfg, cfg)
    return args

def get_transform(dataset_name="Imagenet", model=None):
    """"
    Gets the transform for a given dataset
    """
    if model in ['RN50', 'ViT-B/32']: # if we are evaluating a clip model we use its transforms
        print("...loading CLIP model")
        net, transform = clip.load(model)
        train_transform = transform
    elif dataset_name == "ColoredMNIST":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])
            ])
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])
        ])
    elif dataset_name == 'PlanesExt':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4673, 0.5171, 0.5658], [0.1940, 0.1892, 0.2241])
            ])
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4673, 0.5171, 0.5658], [0.1940, 0.1892, 0.2241])
        ])
    elif dataset_name == "CatsDogs":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5147, 0.4716, 0.4201], [0.2438, 0.2358, 0.2436])
            ])
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5147, 0.4716, 0.4201], [0.2438, 0.2358, 0.2436])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return train_transform, transform

def get_dataset(dataset_name, transform, val_transform=None, biased_val=True):
    if val_transform == None:
        val_transform = transform
    if dataset_name == 'Waterbirds':
        args = get_config('Waterbirds')
        trainset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, transform=transform, biased_val=biased_val)
        valset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'Waterbirds95':
        args = get_config('Waterbirds95')
        trainset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, transform=transform, biased_val=biased_val)
        valset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'WaterbirdsTiny':
        args = get_config('WaterbirdsTiny')
        trainset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        valset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=val_transform)
        testset = WaterbirdsOrig('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=val_transform)
    elif dataset_name == 'WaterbirdsSimple':
        args = get_config('Waterbirds')
        trainset = WaterbirdsSimple('/shared/lisabdunlap/vl-attention/data', args, transform=transform)
        testset = WaterbirdsSimple('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=val_transform)
        valset = testset
    elif dataset_name == 'Planes':
        args = get_config('Planes')
        args.DATA.BIAS_TYPE = 'bias_A'
        trainset = PlanesOrig('/shared/lisabdunlap/vl-attention/data', args, transform=transform, biased_val=biased_val)
        valset = PlanesOrig('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = PlanesOrig('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'PlanesBalanced':
        args = get_config('Planes')
        args.DATA.BIAS_TYPE = 'balanced'
        trainset = PlanesOrig('/shared/lisabdunlap/vl-attention/data', args, transform=transform, biased_val=biased_val)
        valset = PlanesOrig('/shared/lisabdunlap/vl-attention/data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = PlanesOrig('/shared/lisabdunlap/vl-attention/data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'PlanesExt':
        args = get_config('PlanesExt')
        trainset = Planes('/shared/lisabdunlap/vl-attention', args, transform=transform, biased_val=biased_val)
        valset = Planes('/shared/lisabdunlap/vl-attention', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = Planes('/shared/lisabdunlap/vl-attention', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'ColoredMNIST':
        args = get_config('ColoredMNIST')
        args.DATA.CONFOUNDING = 1.0
        trainset = ColoredMNIST('./data/ColoredMNIST', args, transform=transform, biased_val=biased_val)
        valset = ColoredMNIST('./data/ColoredMNIST', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = ColoredMNIST('./data/ColoredMNIST', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'ColoredMNISTBinary':
        args = get_config('ColoredMNIST')
        args.DATA.CONFOUNDING = 1.0
        args.DATA.BIAS_TYPE = 'bin_blue'
        trainset = ColoredMNISTSimplified('./data', args, transform=transform, biased_val=biased_val)
        valset = ColoredMNISTSimplified('./data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = ColoredMNISTSimplified('./data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'MNIST':
        args = None
        trainset = MNIST('./data', args, transform=transform, biased_val=biased_val)
        valset = MNIST('./data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = MNIST('./data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'SVHN':
        args = None
        trainset = SVHN('./data', args, transform=transform, biased_val=biased_val)
        valset = SVHN('./data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = SVHN('./data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'MNIST_SVHN':
        args = None
        trainset = MNIST('./data', args, transform=transform, biased_val=biased_val)
        valset = MNIST('./data', args, split='test', transform=val_transform, biased_val=biased_val)
        testset = SVHN('./data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'ColoredMNISTQuinque':
        args = get_config('ColoredMNIST')
        args.DATA.BIAS_TYPE = 'quinque'
        args.DATA.CONFOUNDING = 1.0
        trainset = ColoredMNISTSimplified('./data', args, transform=transform, biased_val=biased_val)
        valset = ColoredMNISTSimplified('./data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = ColoredMNISTSimplified('./data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == 'ColoredMNISTQuinque95':
        args = get_config('ColoredMNIST')
        args.DATA.BIAS_TYPE = 'quinque'
        args.DATA.CONFOUNDING = 0.95
        trainset = ColoredMNISTSimplified('./data', args, transform=transform, biased_val=biased_val)
        valset = ColoredMNISTSimplified('./data', args, split='val', transform=val_transform, biased_val=biased_val)
        testset = ColoredMNISTSimplified('./data', args, split='test', transform=val_transform, biased_val=biased_val)
    elif dataset_name == "CatsDogs":
        args = get_config('CatsDogs')
        trainset = CatsDogs('./data', args, transform=transform)
        valset = CatsDogs('./data', args, split='val', transform=val_transform)
        testset = CatsDogs('./data', args, split='test', transform=val_transform)
    elif dataset_name == "Imagenet":
        args = get_config('Imagenet')
        trainset = Imagenet('./data', args, transform=transform)
        valset = Imagenet('./data', args, split='val', transform=val_transform)
        testset = Imagenet('./data', args, split='test', transform=val_transform)
    elif dataset_name == "Imagenet-a":
        args = get_config('Imagenet')
        args.DATA.SPLIT='imagenet-a'
        trainset = Imagenet('./data', args, transform=transform)
        valset = Imagenet('./data', args, split='val', transform=val_transform)
        testset = Imagenet('./data', args, split='test', transform=val_transform)
    elif dataset_name == "DomainNet":
        cfg = get_config('DomainNet')
        trainset = DomainNet('/shared/lisabdunlap/data', cfg, split='train', transform=transform)
        valset = DomainNet('/shared/lisabdunlap/data', cfg, split='val', transform=val_transform)
        testset = DomainNet('/shared/lisabdunlap/data', cfg, split='test', transform=val_transform)
    elif dataset_name == "DomainNetMini":
        trainset = OneDomain('/shared/lisabdunlap/data', domain='sketch', split='train', transform=transform)
        valset = OneDomain('/shared/lisabdunlap/data', domain='sketch', split='val', transform=val_transform)
        testset = MultiDomain('/shared/lisabdunlap/data', domains=['clipart', 'painting', 'real'], split='test', transform=val_transform)
    elif dataset_name == "DomainNetMiniReal":
        trainset = OneDomain('/shared/lisabdunlap/data', domain='real', split='train', transform=transform)
        valset = OneDomain('/shared/lisabdunlap/data', domain='real', split='val', transform=val_transform)
        testset = MultiDomain('/shared/lisabdunlap/data', domains=['clipart', 'painting', 'sketch'], split='test', transform=val_transform)
    elif dataset_name == "DomainNetMiniAug":
        trainset = DomainNetMiniAug('/shared/lisabdunlap/data/domainnet_aug', domain='sketch', split='train', transform=transform)
        valset = OneDomain('/shared/lisabdunlap/data', domain='sketch', split='val', transform=val_transform)
        testset = MultiDomain('/shared/lisabdunlap/data', domains=['clipart', 'painting', 'real'], split='test', transform=val_transform)
    elif dataset_name == "DomainNetMiniOracle":
        trainset = MultiDomain('/shared/lisabdunlap/data', domains=['sketch', 'clipart', 'painting', 'real'], split='train', transform=transform)
        valset = MultiDomain('/shared/lisabdunlap/data', domains=['sketch', 'clipart', 'painting', 'real'], split='val', transform=val_transform)
        testset = MultiDomain('/shared/lisabdunlap/data', domains=['sketch', 'clipart', 'painting', 'real'], split='test', transform=val_transform)
    elif dataset_name == "MetaShiftTextile":
        trainset = MetaShiftAttribute(root = '/shared/lisabdunlap/data/allImages/images', classes='textile', transform=transform)
        valset = MetaShiftAttribute(root = '/shared/lisabdunlap/data/allImages/images', classes='textile', transform=val_transform)
        testset = MetaShiftAttribute(root = '/shared/lisabdunlap/data/allImages/images', classes='textile', transform=val_transform)
    elif dataset_name == "CUB":
        trainset = Cub2011('/shared/lisabdunlap/data', train=True, transform=transform)
        valset = Cub2011('/shared/lisabdunlap/data', train=False, transform=val_transform)
        testset = Cub2011Painting('/shared/lisabdunlap/data/CUB-200-Painting', transform=val_transform)
    else:
        raise ValueError(f"{dataset_name} Dataset not supported")

    return trainset, valset, testset

DATASET_CLASSES = {
    "Waterbirds": ['landbird', 'waterbird'],
    "Waterbirds95": ['landbird', 'waterbird'],
    "Planes": ['airbus', 'boeing'],
    "ColoredMNIST": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "CatsDogs": ['cat', 'dog'],
    "PlanesExt": ['airbus', 'boeing'],
    "PlanesBalanced": ['airbus', 'boeing'],
    "ColoredMNISTBinary": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "ColoredMNISTQuinque": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "MNIST": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "SVHN": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "MNIST_SVHN": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "Imagenet": imagenet_classes,
    "Imagenet-a": imagenet_classes,
    # "Living17": LIVING17_CLASSES,
    "DomainNet": DOMAINNET_CLASSES,
    "DomainNetMini": MINI_DOMAINNET_CLASSES,
    "DomainNetMiniAug": MINI_DOMAINNET_CLASSES,
    "DomainNetMiniOracle": MINI_DOMAINNET_CLASSES,
    "CUB": CUB_CLASSES,
}

DATASET_DOMAINS = {
    "Waterbirds": ['land', 'water'],
    "Waterbirds95": ['land', 'water'],
    "Planes": ['grass', 'road'],
    "PlanesExt": ['grass', 'road'],
    "PlanesBalanced": ['grass', 'road'],
    "ColoredMNISTBinary": ['red', 'blue'],
    "ColoredMNISTQuinque": ['red','green','yellow','pink','blue'],
    "DomainNet": ['photo', 'sketch'],
    "DomainNetMini": MINI_DOMAINS,
    "DomainNetMiniAug": MINI_DOMAINS,
    "DomainNetMiniOracle": MINI_DOMAINS,
    "CUB": CUB_DOMAINS,
    "SVHN": ["MNIST", "SVHN"],
    "MNIST_SVHN": ["MNIST", "SVHN"]
}

def get_domain(dataset_name):
    return DATASET_DOMAINS[dataset_name]

def get_class(dataset_name):
    return DATASET_CLASSES[dataset_name]

def get_cache_file(dataset_name, model_name='ViT-B/32', biased_val=True, model_type='clip'):
    if biased_val:
        assert dataset_name in BIASED_DATASET_PATHS[model_type][model_name].keys(), f"{dataset_name} is not cached or not added to the DATASET_PATHS dict in helpers/dataset_helpers.py"
        return BIASED_DATASET_PATHS[model_type][model_name][dataset_name], DATASET_CLASSES[dataset_name], DATASET_DOMAINS[dataset_name]
    else:
        assert dataset_name in DATASET_PATHS[model_name].keys(), f"{dataset_name} is not cached or not added to the DATASET_PATHS dict in helpers/dataset_helpers.py"
        return DATASET_PATHS[model_name][dataset_name], DATASET_CLASSES[dataset_name], DATASET_DOMAINS[dataset_name]