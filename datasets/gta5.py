import os
from PIL import Image
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.gta_labels import GTA5Labels_TaskCV2017

# TODO: implement here your custom dataset class for GTA5


class GTA5(Dataset):
    
    label_map = GTA5Labels_TaskCV2017().list_
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = GTA5Labels_TaskCV2017().list_
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.label_dir = os.path.join(self.root_dir, 'labels')
        self.images = os.listdir(self.image_dir) 
        
    def __len__(self):
        return len(self.images)

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=self.label_map)

    @staticmethod
    def _decode(lbl, label_map):
        color_lbl = np.zeros((*lbl.shape, 3))
        for label in label_map:
            color_lbl[lbl == label.ID] = label.color
        return color_lbl        
    
    def __getitem__(self, idx): 
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx].replace('images', 'labels'))
        
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(np.array(Image.open(label_name)))
        color_lbl = self._decode(label, self.label_map) # adjusting label format to be equal the one from cityscapes
        
        if self.transform:
            image = self.transform[0](image)
            color_lbl = self.transform[1](color_lbl)
        
        return image, color_lbl
    

# references:
# https://github.com/sarrrrry/PyTorchDL_GTA5/blob/master/tests/test_dataloader.py