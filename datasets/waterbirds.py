import torch
from torchvision import transforms
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image

from utils import get_counts

GROUP_NAMES = np.array(['Land_on_Land', 'Land_on_Water', 'Water_on_Land', 'Water_on_Water'])

def get_label_mapping():
    return np.array(['Landbird', 'Waterbird'])

class WaterbirdsOrig(torch.utils.data.Dataset):
    def __init__(self, root, cfg, split='train', transform=None, biased_val=False):
        self.cfg = cfg
        # self.original_root       = os.path.expanduser(root)
        self.original_root = root
        self.transform  = transform
        self.split      = split
        self.root       = os.path.join(self.original_root, cfg.DATA.WATERBIRDS_DIR)
        self.return_seg = True
        self.return_bbox = False
        self.size       = cfg.DATA.SIZE

        print('WATERBIRDS DIR: {}'.format(self.root))

        self.seg_transform = transforms.Compose([
            transforms.Resize((self.size,self.size)),
            transforms.ToTensor(),
        ])

        # metadata
        # self.metadata_df = pd.read_csv(
        #     os.path.join(self.root, 'metadata.csv'))
        self.metadata_df = pd.read_csv(cfg.DATA.METADATA)

        # Get the y values
        self.labels = self.metadata_df['y'].values
        self.num_classes = 2
        self.class_labels = ['Landbird', 'Waterbird']

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.labels*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # self.seg_data  =  np.array([os.path.join(root, 'CUB_200_2011/segmentations',
        #                                          path.replace('.jpg', '.png')) for path in self.filename_array])

        self.data = np.array([os.path.join(self.root, filename) for filename in self.filename_array])

        if self.cfg.DATA.BLUR_ATTENTION:
            self.attention_data = np.array([os.path.join(self.root, cfg.DATA.ATTENTION_DIR,
                                                path.replace('.jpg', '.pth')) for path in self.filename_array])

        mask = self.split_array == self.split_dict[self.split]
        num_split = np.sum(mask)
        self.indices = np.where(mask)[0]

        self.labels = torch.Tensor(self.labels)
        self.group_array = torch.Tensor(self.group_array)

        # Arrays holding image filenames and labels for just the split, not all data.
        # Useful for detection approach to quickly access labels & filenames
        self.image_filenames     = []
        self.labels_split        = []
        self.group_labels_split  = []
        self.confounder_split    = []
        for idx in self.indices:
            if biased_val and split == 'val':
                if self.group_array[idx].item() in [0, 3]:
                    self.image_filenames.append(self.data[idx])
                    self.labels_split.append(self.labels[idx])
                    self.group_labels_split.append(self.group_array[idx])
                    self.confounder_split.append(self.confounder_array[idx])
            else:
                self.image_filenames.append(self.data[idx])
                self.labels_split.append(self.labels[idx])
                self.group_labels_split.append(self.group_array[idx])
                self.confounder_split.append(self.confounder_array[idx])
        self.image_filenames    = np.array(self.image_filenames)
        self.labels_split       = torch.Tensor(self.labels_split)
        self.group_labels_split = torch.Tensor(self.group_labels_split)
        self.confounder_split   = torch.Tensor(self.confounder_split)
        self.class_weights = get_counts(self.labels_split)


        print('NUMBER OF SAMPLES WITH LABEL {}: {}'.format(get_label_mapping()[0],
                                                           len(torch.where(self.labels_split == 0)[0]))
        )
        print('NUMBER OF SAMPLES WITH LABEL {}: {}'.format(get_label_mapping()[1],
                                                           len(torch.where(self.labels_split == 1)[0]))
        )

        for i in range(len(GROUP_NAMES)):
            print('NUMBER OF SAMPLES WITH GROUP {}: {}'.format(GROUP_NAMES[i],
                                                               len(torch.where(self.group_labels_split == i)[0]))
            )

    def create_subset(self):
        subset_size = self.cfg.DATA.SUBSET_SIZE
        images_per_class = subset_size // 2
        inds = {
            'class_0': torch.where(self.labels[self.indices] == 0)[0],
            'class_1': torch.where(self.labels[self.indices] == 1)[0]
        }

    def get_filenames(self, indices):
        """
        Return list of filenames for requested indices.
        Need to access self.indices to map the requested indices to the right ones in self.data
        """
        filenames = []
        for i in indices:
            new_index = self.indices[i]
            filenames.append(self.data[new_index])
        return filenames

    def get_label(self, filename):
        return self.labels[np.where(self.data == filename)[0]]

    def __getitem__(self, index):

        path = self.image_filenames[index]
        label = self.labels_split[index]
        group = self.group_labels_split[index]
        place = self.confounder_split[index]

        # group = torch.Tensor([group])

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.cfg.DATA.BLUR_ATTENTION:
            att = torch.load(self.attention_data[index])
            if 'deeplab' in self.cfg.DATA.ATTENTION_DIR:
                att = att['mask']
            else:
                print(att.keys())
                att = att['attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear',
                                align_corners=False)#[0]
            att = torch.mean(att, dim=0)
            att -= torch.min(att)
            att /= torch.max(att)
            print(torch.min(att), torch.max(att))
            att = torch.squeeze(torch.stack([att, att, att]), dim=1)
            null_img = torch.zeros(att.shape)

            img = torch.where(att > self.cfg.DATA.ATTN_THRESHOLD, img, null_img)

        return {
            'image_path': path,
            'image': img,
            'label': int(label),
            'group': group,
            'domain': place,
            'filename': path
        }
        # return img, label


    def __len__(self):
        return len(self.labels_split)

class Waterbirds(WaterbirdsOrig):
    
    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        return results['image'], results['label']

class WaterbirdsBoring(WaterbirdsOrig):

    def __init__(self, root, cfg, split='train', transform=None, 
                    land_background='/home/lisabdunlap/EditingClassifiers/data/waterbirds/forest.jpg', 
                    water_background='/home/lisabdunlap/EditingClassifiers/data/waterbirds/ocean_2.jpg'):
        super().__init__(root, cfg, split, transform, metadata)
        self.land_background = land_background
        self.water_background = water_background
    
    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        new_results = results.copy()
        mask_copy = np.ones(results['seg'].shape)
        train_masks = np.logical_xor(mask_copy, results['seg'])
        # place =  0 -> land place = 1 -> water
        pattern_img_path = self.land_background if results['place'] == 1 else self.water_background
        if self.transform:
            pattern_img = self.transform(Image.open(pattern_img_path))[:3, :, :]
            modified_img = results['image'] * (1-train_masks) + pattern_img * train_masks
        else:
            pattern_img = Image.open(pattern_img_path)
            modified_img = results['image'] * (1-train_masks) + pattern_img * train_masks
        
        return {
            'img': modified_img, 
            'mask': train_masks, 
            'label': results['label']
            }

class WaterbirdsSimple(WaterbirdsBoring):

    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        return results['img'], results['label']


class DrawRect(object):
    def __init__(self, start_pos=(20,20), width=32, color=(255,0,0)):
        self.start_pos = start_pos
        self.end_pos   = (start_pos[0] + width, start_pos[1] + width)
        self.color     = color
    def __call__(self, img):
        """ Draw rectangle on PIL Image, return PIL Image"""
        img = np.array(img)
        img[self.start_pos[0]:self.end_pos[0], self.start_pos[0]:self.end_pos[1], :] = self.color
        #img = cv2.rectangle(np.array(img), self.start_pos, self.end_pos, self.color, -1)
        img = Image.fromarray(img)
        return img


def get_loss_upweights(bias_fraction=0.95, mode='per_class'):
    """
    For weighting training loss for imbalanced classes.

    Returns 1D tensor of length 2, with loss rescaling weights.

    weight w_c for class c in C is calculated as:
    (1 / num_samples_in_class_c) / (max(1/num_samples_in_c) for c in num_classes)

    """
    assert mode in ['per_class', 'per_group']

    # Map bias fraction to per-class and per-group stats.
    training_dataset_stats = {
        0.95: {
            'per_class': [3682, 1113],
            'per_group': [3498, 184, 56, 1057]
        },
        1.0: {
            'per_class': [3694, 1101]
        }
    }
    counts  = training_dataset_stats[bias_fraction][mode]
    counts  = torch.Tensor(counts)
    fracs   = 1 / counts
    weights = fracs / torch.max(fracs)

    return weights

# class WaterbirdsEditing(WaterbirdsOrig):

#     def __init__(self, root, cfg, split='train', transform=None):
#         super().__init__(root, cfg, split, transform, metadata='/home/lisabdunlap/EditingClassifiers/data/waterbirds/land_birds.csv')
#         # df = pd.read_csv('/home/lisabdunlap/EditingClassifiers/data/waterbirds/metadata.csv')
#         # df['img_id']

#     def __getitem__(self, index):

#         results = super().__getitem__(index)
#         return {
#             'img': results['image'], 
#             'mask': results['seg'], 
#             'label': results['label']
#             }