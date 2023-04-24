import nibabel as nib
from pathlib import Path
import os 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import imgaug
import torch
import imgaug.augmenters as aai
from imgaug.augmentables.segmaps import SegmentationMapsOnImag

class Left_DataCostume(Dataset):
    def __init__(self, root_path: str, augment_params):
        self.root_list = self.extract_file(root_path)
        self.augment_params = augment_params
        
    @staticmethod
    def extract_file(root_list):
        file_path=[]
        for sub_path in root_list.glob("*"):
            data_file= sub_path/"data"
            for file in data_file.glob("*.npy"):
                file_path.append(file)
        return file_path 
    @staticmethod
    def change_img_to_lab(file_):
        parts = list(file_.parts)
        parts[parts.index("data")] = "mask"
        return Path(*parts) 
    
    def augment(self,image, mask):
        rand_seed = torch.randint(0,100000,(1,)).item()
        imgaug.seed(rand_seed)
        mask = SegmentationMapsOnImage(mask, mask.shape)
        image_aug , mask_aug = self.augment_params(image=image, segmentation_maps = mask)
        mask_aug = mask_aug.get_arr()
        return image_aug,mask_aug
        
    def __len__(self):
        return len(self.root_list)
    
    def __getitem__(self,idx):
        image_path = self.root_list[idx]
        label_path = self.change_img_to_lab(image_path)
        image = np.load(image_path).astype(np.float32)
        mask = np.load(label_path)
        if self.augment_params:
            image , mask = self.augment(image,mask)
        return np.expand_dims(image,0),np.expand_dims(mask,0)   
