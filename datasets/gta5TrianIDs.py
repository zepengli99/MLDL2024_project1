import os
import numpy as np
from PIL import Image
from gta5Labels import GTA5Labels_TaskCV2017

# This class is created to convert GTA5 color labels to trainIDs labels
class GTA5TrainID:
    label_map = GTA5Labels_TaskCV2017().trans_label_map()
    LBL_DIR_NAME = "labels"
    TRAINIDS_DIR_NAME = "trainIdsLabels"
    
    
    def __init__(self, root):
        self.root = root
        self.lbl_paths = self.create_lblpath_list()
        # print(self.label_map)

    def create_lblpath_list(self):
        lbl_dir = os.path.join(self.root, self.LBL_DIR_NAME)
        lbl_paths = os.listdir(lbl_dir)
        return lbl_paths
    
    def labelToTrainIDs(self, idx):
        lbl_path = os.path.join(self.root, self.LBL_DIR_NAME, self.lbl_paths[idx])
        out_put_dir = os.path.join(self.root, self.TRAINIDS_DIR_NAME)
        if not os.path.exists(out_put_dir):
            os.makedirs(out_put_dir)
        out_put_path = os.path.join(self.root, self.TRAINIDS_DIR_NAME, self.lbl_paths[idx])
        if os.path.exists(out_put_path):
            print(out_put_path +" has already existed.")
        else:  
            img_trainIDs = self.decode(self.read_img(lbl_path))
            img_labelTrianIDs = Image.fromarray(img_trainIDs.astype(np.uint8))
            img_labelTrianIDs.save(out_put_path)
            print(out_put_path +" is generated")

    @staticmethod
    def read_img(path):
        img = Image.open(str(path)).convert('RGB')
        img = np.array(img)
        return img

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map)

    @staticmethod
    def _decode(lbl, label_map):
        def to_trainIDs(x, label_map=label_map):
            return label_map.get(tuple(x), 255)
        return np.apply_along_axis(to_trainIDs, axis=2, arr=lbl)
     
    
if __name__ == '__main__':
    trainIds = GTA5TrainID('datasets\\GTA5')
    # trainIds.labelToTrainIDs(0)
    for i in range(len(trainIds.lbl_paths)):
        trainIds.labelToTrainIDs(i)