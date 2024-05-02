import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets.gta_labels import GTA5Labels_TaskCV2017


class GTA5(Dataset):
    
    label_map = GTA5Labels_TaskCV2017().trans_label_map()

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

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = self.PathPair_ImgAndLabel(root=self.root)

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
            img = self.transform[0](img)
            tid = self.transform[1](tid)
            
        return img, tid
# references:
# https://github.com/sarrrrry/PyTorchDL_GTA5/blob/master/tests/test_dataloader.py