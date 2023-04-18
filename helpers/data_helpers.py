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
from datasets.colored_mnist import ColoredMNIST, ColoredMNISTSimplified, MNIST, SVHN
from datasets.domain_net import DomainNet, DOMAINNET_CLASSES, MINI_DOMAINNET_CLASSES, MINI_DOMAINS, OneDomain, MultiDomain, DomainNetMiniAug
from datasets.cub import Cub2011Painting, Cub2011, CUB_DOMAINS, CUB_CLASSES
from datasets.office_home import OfficeHome, OFFICE_HOME_CLASSES, OFFICE_HOME_DOMAINS
from datasets.grozi import Products, GROZI_CLASSES, GROZI_DOMAINS

def get_config(name="Waterbirds"):
    base_cfg  = OmegaConf.load('data_configs/base.yaml')
    if name == "Waterbirds":
        cfg       = OmegaConf.load('data_configs/waterbirds.yaml')
    elif "ColoredMNIST" in name:
        cfg       = OmegaConf.load('data_configs/colored_mnist.yaml')
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

def get_dataset(dataset_name, root, transform, val_transform=None):
    if val_transform == None:
        val_transform = transform
    if dataset_name == 'Waterbirds':
        args = get_config('Waterbirds')
        trainset = WaterbirdsOrig(root, args, transform=transform)
        valset = WaterbirdsOrig(root, args, split='val', transform=val_transform)
        testset = WaterbirdsOrig(root, args, split='test', transform=val_transform)
    elif dataset_name == 'Waterbirds95':
        args = get_config('Waterbirds95')
        trainset = WaterbirdsOrig(root, args, transform=transform)
        valset = WaterbirdsOrig(root, args, split='val', transform=val_transform)
        testset = WaterbirdsOrig(root, args, split='test', transform=val_transform)
    elif dataset_name == 'ColoredMNISTBinary':
        args = get_config('ColoredMNIST')
        args.DATA.CONFOUNDING = 1.0
        args.DATA.BIAS_TYPE = 'bin_blue'
        trainset = ColoredMNISTSimplified('./data', args, transform=transform)
        valset = ColoredMNISTSimplified('./data', args, split='val', transform=val_transform)
        testset = ColoredMNISTSimplified('./data', args, split='test', transform=val_transform)
    elif dataset_name == 'MNIST':
        args = None
        trainset = MNIST('./data', args, transform=transform)
        valset = MNIST('./data', args, split='val', transform=val_transform)
        testset = MNIST('./data', args, split='test', transform=val_transform)
    elif dataset_name == 'SVHN':
        args = None
        trainset = SVHN('./data', args, transform=transform)
        valset = SVHN('./data', args, split='val', transform=val_transform)
        testset = SVHN('./data', args, split='test', transform=val_transform)
    elif dataset_name == 'MNIST_SVHN':
        args = None
        trainset = MNIST('./data', args, transform=transform)
        valset = MNIST('./data', args, split='test', transform=val_transform)
        testset = SVHN('./data', args, split='test', transform=val_transform)
    elif dataset_name == "DomainNet":
        cfg = get_config('DomainNet')
        trainset = DomainNet(root, cfg, split='train', transform=transform)
        valset = DomainNet(root, cfg, split='val', transform=val_transform)
        testset = DomainNet(root, cfg, split='test', transform=val_transform)
    elif dataset_name == "DomainNetMini":
        trainset = OneDomain(root, domain='sketch', split='train', transform=transform)
        valset = OneDomain(root, domain='sketch', split='val', transform=val_transform)
        testset = MultiDomain(root, domains=['clipart', 'painting', 'real'], split='test', transform=val_transform)
    elif dataset_name == "DomainNetMiniReal":
        trainset = OneDomain(root, domain='real', split='train', transform=transform)
        valset = OneDomain(root, domain='real', split='val', transform=val_transform)
        testset = MultiDomain(root, domains=['clipart', 'painting', 'sketch'], split='test', transform=val_transform)
    elif dataset_name == "DomainNetMiniOracle":
        trainset = MultiDomain(root, domains=['sketch', 'clipart', 'painting', 'real'], split='train', transform=transform)
        valset = MultiDomain(root, domains=['sketch', 'clipart', 'painting', 'real'], split='val', transform=val_transform)
        testset = MultiDomain(root, domains=['sketch', 'clipart', 'painting', 'real'], split='test', transform=val_transform)
    elif dataset_name == "CUB":
        trainset = Cub2011(root, train=True, transform=transform)
        valset = Cub2011(root, train=False, transform=val_transform)
        testset = Cub2011Painting('/shared/lisabdunlap/data/CUB-200-Painting', transform=val_transform)
    elif dataset_name == "OfficeHomeProduct":
        trainset = OfficeHome(root, domains=["Product"], train=True, transform=transform)
        valset = OfficeHome(root, domains=["Product"], train=False, transform=transform)
        testset = OfficeHome(root, domains=["Art", "Clipart", "Real World"],train=True, transform=transform)
    elif dataset_name == "OfficeHomeClipart":
        trainset = OfficeHome(root, domains=["Clipart"], train=True, transform=transform)
        valset = OfficeHome(root, domains=["Clipart"], train=False, transform=transform)
        testset = OfficeHome(root, domains=["Product", "Art", "Real World"],train=True, transform=transform)
    elif dataset_name == "OfficeHomeArt":
        trainset = OfficeHome(root, domains=["Art"], train=True, transform=transform)
        valset = OfficeHome(root, domains=["Art"], train=False, transform=transform)
        testset = OfficeHome(root, domains=["Clipart", "Product", "Real World"],train=True, transform=transform)
    else:
        raise ValueError(f"{dataset_name} Dataset not supported")

    return trainset, valset, testset

DATASET_CLASSES = {
    "Waterbirds": ['landbird', 'waterbird'],
    "Waterbirds95": ['landbird', 'waterbird'],
    "ColoredMNISTBinary": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "MNIST": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "SVHN": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "MNIST_SVHN": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    "DomainNet": DOMAINNET_CLASSES,
    "DomainNetMini": MINI_DOMAINNET_CLASSES,
    "DomainNetMiniOracle": MINI_DOMAINNET_CLASSES,
    "CUB": CUB_CLASSES,
    "OfficeHomeProduct": OFFICE_HOME_CLASSES,
    "OfficeHomeClipart": OFFICE_HOME_CLASSES,
    "OfficeHomeArt": OFFICE_HOME_CLASSES,
}

DATASET_DOMAINS = {
    "Waterbirds": ['forest', 'water'],
    "Waterbirds95": ['forest', 'water'],
    "ColoredMNISTBinary": ['red', 'blue'],
    "DomainNetMini": MINI_DOMAINS,
    "DomainNetMiniOracle": MINI_DOMAINS,
    "CUB": CUB_DOMAINS,
    "SVHN": ["MNIST", "SVHN"],
    "MNIST_SVHN": ["MNIST", "SVHN"],
    "OfficeHomeProduct": OFFICE_HOME_DOMAINS,
    "OfficeHomeClipart": OFFICE_HOME_DOMAINS,
    "OfficeHomeArt": OFFICE_HOME_DOMAINS,
}

def get_domain(dataset_name):
    return DATASET_DOMAINS[dataset_name]

def get_class(dataset_name):
    return DATASET_CLASSES[dataset_name]