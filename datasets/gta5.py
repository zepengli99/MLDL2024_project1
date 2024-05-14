import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from gta5Labels import GTA5Labels_TaskCV2017
from torchvision.transforms import functional as F
from torchvision import transforms


class GTA5(Dataset):

    class PathPair_ImgAndLabel:
        IMG_DIR_NAME = "images"
        TRAINIDS_DIR_NAME = "trainIdsLabels"

        def __init__(self, root):
            self.root = root
            self.img_paths = self.create_imgpath_list()
            self.tid_paths = self.create_tidpath_list()

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            tid_path = self.tid_paths[idx]
            return img_path, tid_path

        def create_imgpath_list(self):
            img_dir = os.path.join(self.root, self.IMG_DIR_NAME)
            img_paths = os.listdir(img_dir)
            return img_paths

        def create_tidpath_list(self):
            tid_dir = os.path.join(self.root, self.TRAINIDS_DIR_NAME)
            tid_paths = os.listdir(tid_dir)
            return tid_paths

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.label_map = GTA5Labels_TaskCV2017()
        self.transform = transform
        self.target_transform = target_transform
        self.paths = self.PathPair_ImgAndLabel(root=self.root)

    def classes_ids(self):
        return self.label_map.support_id_list

    def classes(self):
        return self.label_map.label_list

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, isPath=False):
        img_path, tid_path = self.paths[idx]
        img_path = os.path.join(self.paths.root, self.paths.IMG_DIR_NAME, img_path)
        tid_path = os.path.join(self.paths.root, self.paths.TRAINIDS_DIR_NAME, tid_path)
        if isPath:
            return img_path, tid_path

        img = Image.open(str(img_path)).convert('RGB')
        tid = Image.open(str(tid_path))

        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            tid = self.target_transform(tid)

        return img, torch.tensor(np.array(tid)).long()
       

# An example about how to use this class to load GTA5 dataset
# transform = transforms.Compose([
#     transforms.Resize((720,1280), interpolation=transforms.InterpolationMode.NEAREST),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# target_transform = transforms.Compose([
#     transforms.Resize((720,1280), interpolation=transforms.InterpolationMode.NEAREST)
# ])

# transform with augmentation
# transform_aug = transforms.Compose([    
#     # flip/rotarion/crop
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=30),
#     transforms.RandomResizedCrop(size=(720,1280), interpolation=transforms.InterpolationMode.NEAREST),

#     # transformation for colorful image
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),

#     # gaussian blur
#     transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
      
#     *transform.transforms

# ])

# transform with augmentation
# target_transform_aug = transforms.Compose([
#     # flip/rotarion/crop
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=30),
#     transforms.RandomResizedCrop(size=(720,1280), interpolation=transforms.InterpolationMode.NEAREST),

#     *target_transform.transforms
# ])

# train_dataset = GTA5('datasets\\GTA5', transform=transform_aug, target_transform=target_transform_aug)