import os
from PIL import Image
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
# TODO: implement here your custom dataset class for Cityscapes


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, 'images', self.split)
        self.label_dir = os.path.join(self.root_dir, 'gtFine', self.split)
        self.image_folders = os.listdir(self.image_dir)
        self.images = []

        for folder in self.image_folders:
            images_in_folder = os.listdir(os.path.join(self.image_dir, folder))
            self.images.extend(images_in_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        folder_name = self.images[idx].split('_')[0]
        img_name = os.path.join(self.image_dir, folder_name, self.images[idx])
        label_name = os.path.join(self.label_dir, folder_name, self.images[idx].replace('leftImg8bit', 'gtFine_labelTrainIds'))

        image = np.array(Image.open(img_name).convert('RGB'))[::2,::2]

        label = np.array(Image.open(label_name))[::2,::2]
        label = torch.tensor(label).long()
        if self.transform[0]:
            image = self.transform[0](image)
        if self.transform[1]:
            label = self.transform[1](label)
        
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be of type torch.Tensor but got {type(image)}")

        return image, label

# An example about how to use this class to load Cityspaces dataset
# import torchvision.transforms as transforms
# transform = transforms.Compose([
#     transforms.Resize((1024,512)) # resize to 1024*512  # need to be updated
#     transforms.ToTensor()
# ])

# You should specify the path of your Cityspaces dataset
# train_dataset = CityscapesDataset('datasets\\Cityspaces', transform=transform) 
