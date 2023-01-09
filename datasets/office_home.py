import torch 
import torchvision
import numpy as np
import os
import pandas as pd

OFFICE_HOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']
OFFICE_HOME_CLASSES = ['Alarm_Clock', 'Backpack', 'Batteries','Bed','Bike','Bottle','Bucket','Calculator','Calendar','Candles','Chair','Clipboards','Computer','Couch','Curtains',
 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives',
 'Lamp_Shade','Laptop','Marker','Monitor','Mop','Mouse','Mug','Notebook','Oven','Pan','Paper_Clip','Pen','Pencil','Postit_Notes','Printer','Push_Pin',
 'Radio','Refrigerator','Ruler','Scissors','Screwdriver','Shelf','Sink','Sneakers','Soda','Speaker','Spoon','TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can','Webcam']

class OfficeHome(torchvision.datasets.ImageFolder):

    def __init__(self, root, domains=['Art'], transform=None, train=True):
        assert all([d in OFFICE_HOME_DOMAINS for d in domains]), f"Domain must be one of {OFFICE_HOME_DOMAINS}"
        global_root = os.path.join(root, 'OfficeHomeDataset_10072016')
        self.all_samples, self.domains = [], []
        for d in domains:
            torchvision.datasets.ImageFolder.__init__(self, root=os.path.join(global_root, d), transform=transform)
            train_samples, val_samples = self.get_splits(self.samples)
            sample = train_samples if train else val_samples
            self.all_samples.extend(sample)
            self.domains.extend([OFFICE_HOME_DOMAINS.index(d)] * len(sample))
        self.samples = self.all_samples
        
    def get_splits(self, samples):
        train_samples, val_samples = [], []
        for c in range(len(self.classes)):
            cls_samples = [s for s in samples if s[1] == c]
            train_samples += cls_samples[:int(len(cls_samples) * 0.8)] # 80% train
            val_samples += cls_samples[int(len(cls_samples) * 0.8):] # 20% val
        return train_samples, val_samples

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]
        domain = self.domains[index]
        return {
            "image": img,
            "label": target,
            "domain": domain,
            "group": domain, # since we dont have group labels for test set
            "filename": path
        }