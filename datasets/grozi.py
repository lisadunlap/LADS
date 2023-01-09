import torch
import numpy as np
from PIL import Image
import os
import glob

GROZI_CLASSES = ['Bausch & Lomb Renu All in One Multi Purpose Solution',
 'Chex Mix',
 "Gardetto's Original Recipe",
 'doublemint gum',
 'big red gum',
 'Vicks DayQuil LiquiCaps',
 'Hershey Milk Chocolate with Almonds',
 'Twix Cookie Bar',
 'CLOROX 2 LIQUID 44 OZ',
 'POCKETCOMB',
 'Motrin IB Ibuprofen Tablets USP',
 'Neosporin Original',
 'Tic Tac Wintergreen',
 'DENTYNE ICE ARTIC CHILL',
 'Genuine BAYER Aspirin, tablets (325 mg)',
 '26OZ BLUE WINDEX',
 'Morton Salt, Iodized',
 'Ziploc Sandwich Bags',
 'Aleve Caplets',
 "Reese's Pieces",
 'Pringles Pizza-licious',
 'Starburst Original Fruit',
 'Skittles (Original)',
 "French's Classic Yellow Mustard",
 'FORMULA 409',
 'Monster Energy Beverage',
 'Tapatio - Salsa Picante',
 'CARMEX EZ-ON APPLICATOR',
 'Nestle Crunch',
 'ARM + HAMMER BAKING SODA',
 'Diet Coke with Lime',
 'Sprite 12oz Can',
 'Dr Pepper',
 'Red Bull Sugarfree',
 'LINDOR CANDY',
 'Haribo Gold-Bears Gummi Candy',
 'TOBLERONE MILK CHOCOLATE',
 'Tylenol  Extra Strength Caplets',
 'David Sunflower Seeds',
 'Always thin pantiliners',
 'Soft Scrub with Bleach Cleanser',
 'Gillette Foamy Shaving Cream, Regular',
 "Campbell's Tomato Soup - Microwavable bowl",
 'Vivarin',
 'KELL RAISIN BRAN 15 OZ 121459',
 'PEPTO BISMOL CHERRY TAB 30 CT',
 'Tabasco Brand Pepper Sauce',
 'Trojan-Enz lubricated condoms',
 'Mountain Dew, Single Bottle',
 'Jif Creamy Peanut Butter',
 'Campbells Cream of Chicken soup',
 'A-1 STEAK SAUCE']

GROZI_DOMAINS = ['inVitro', 'inSitu']

class Products:

    def __init__(self, root='/shared/lisabdunlap/data/products', transform=None, split='train'):
        self.root = root
        self.transform = transform
        self.split = split
        if split == 'test':
            self.root = os.path.join(self.root, 'inSitu')
        else:
            self.root = os.path.join(self.root, 'inVitro')

        with open('/shared/lisabdunlap/data/products/invitro_classes.txt') as f:
            lines = f.readlines()
        classes = [l.strip() for l in lines if len(l.strip()) > 0][3::3]
        class_idxs = [classes.index(c) for c in GROZI_CLASSES]
        self.samples = []
        self.classes = []
        self.class_names = []
        lab = 0
        for index in range(1, 121):
            if split == 'test':
                items = [i for i in glob.glob(f'{self.root}/{index}/video/*') if '.png' in i]
            else:
                items = [i for i in glob.glob(f'{self.root}/{index}/*/JPEG/*') if '.jpg' in i]
            if index-1 in class_idxs:
                if split == 'train':
                    self.samples.extend([(lab, i) for i in items[:5]])
                elif split == 'val':
                    self.samples.extend([(lab, i) for i in items[5:]])
                else:
                    self.samples.extend([(lab, i) for i in items])
                self.classes.append(lab)
                self.class_names.append(classes[index-1])
                lab += 1
    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def __getitem__(self, idx):
        label, path = self.samples[idx]
        img = Image.open(path)
        #add white background
        try:
            img = self.expand2square(img, (255, 255, 255)).resize((224, 224))
        except:
            print("ERROR ", path)
        if self.transform:
            img = self.transform(img)
        # return img, label
        return {
            "image": img,
            "label": label,
            "group": 0,
            "domain": 1 if self.split == 'test' else 0,
            "filename": path,
        }