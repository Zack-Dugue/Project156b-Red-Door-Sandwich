import torch
import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader,Dataset
import os
import random
# import cv2
from PIL import Image
from torchvision.io import read_image

# from skimage import io


class LungSegDataset(Dataset): # inherit from torch.utils.data.Dataset
    "Lung sengmentation dataset."
    def __init__(self,root_dir = os.path.join(os.getcwd(),"data/Lung Segmentation"),split = "train", transforms = None):
        """
        Args:
        :param root_dir (str):
        :param split (str):
        :param transforms (callable, optional) :
        """
        self.root_dir = root_dir
        self.split = split # train / val / test
        self.transforms = transforms

        # data
        # train set : CHN
        # test/validation set : MCU
        self.image_path = self.root_dir + '/CXR_png'
        image_file =  os.listdir(self.image_path)
        self.train_image_file = [fName for fName in image_file if "CHNCXR" in fName]
        self.train_image_idx = sorted([int(fName.split("_")[1]) for fName in self.train_image_file])

        self.eval_image_file = [fName for fName in image_file if "MCUCXR" in fName]
        self.eval_image_idx = sorted([int(fName.split("_")[1]) for fName in self.eval_image_file])

        # target
        self.mask_path = os.path.join(self.root_dir,'masks')
        mask_file = os.listdir(self.mask_path)
        self.train_mask_file = [fName for fName in mask_file if "CHNCXR" in fName]
        self.train_mask_idx = sorted([int(fName.split("_")[1]) for fName in self.train_mask_file])

        self.eval_mask_file = [fName for fName in mask_file if "MCUCXR" in fName]
        self.eval_mask_idx = sorted([int(fName.split("_")[1]) for fName in self.eval_mask_file])

        # train/ val / test
        # for train set, we use CHN
        # for test and validation set, we use MCU
        self.train_idx = [idx for idx in self.train_image_idx if idx in self.train_mask_idx]
        self.eval_idx = [idx for idx in self.eval_image_idx if idx in self.eval_mask_idx]
        self.val_idx = self.eval_idx[:int(0.5*len(self.eval_idx))]
        self.test_idx = self.eval_idx[int(0.5*len(self.eval_idx)):]




        self.data_file = {"train"  : {"image":self.train_image_file , "mask": self.train_mask_file},
                           "val"   : {"image":self.eval_image_file  , "mask": self.eval_mask_file },
                           "test"  : {"image":self.eval_image_file  , "mask": self.eval_mask_file}}

        self.data_idx ={"train" : self.train_idx,
                        "val"   : self.val_idx,
                        "test"  : self.test_idx}



        # print("The Total number of data =",len(self.train_idx) + len(self.val_idx) + len(self.test_idx))
        # print("The Total number of train data =", len(self.train_idx))
        # print("The Total number of val data =", len(self.val_idx))
        # print("The Total number of test data =", len(self.test_idx))


    def __len__(self):
        return len(self.data_idx[self.split])

    def __getitem__(self, idx):
        idx = self.data_idx[self.split][idx]
        # set index
        for fName in self.data_file[self.split]["image"]:
            file_idx = int(fName.split('_')[1])
            if idx == file_idx:
                img_fName = fName
        img_path = os.path.join(self.image_path, img_fName)
        img = read_image(img_path).to(torch.float32) # open as PIL Image and set Channel = 1
        img = tv.transforms.Grayscale(1)(img)
        # img = cv2.imread(img_path)

        for fName in self.data_file[self.split]["mask"]:
            file_idx = int(fName.split('_')[1])
            if idx == file_idx:
                mask_fName = fName
        mask_path = os.path.join(self.mask_path, mask_fName)
        mask = read_image(mask_path).to(torch.float32)  # PIL Image
        # mask = cv2.imread(mask_path)

        if self.transforms:
            img = self.transforms(img)
            # print(img.size())
            mask = self.transforms(mask)
            # print(mask.size())
        sample = {'image': img, 'mask': mask}

        # if self.transforms:
        #     sample = self.transforms(sample)

        if isinstance(img,torch.Tensor) and isinstance(mask, torch.Tensor):
            assert img.size() == mask.size()
        return sample
    
def make_dataloader(batch_size, num_dataloaders, split = 'train', img_size=512):
    if split == 'train':
        transforms = tv.transforms.Compose([
            # tv.transforms.Grayscale(),
            tv.transforms.Resize((img_size, img_size), interpolation= tv.transforms.InterpolationMode.BILINEAR, antialias=None)
        ])
        dataset = LungSegDataset(split="train", transforms=transforms)
        shuffle = True
        
    if split == 'test':
        transforms = tv.transforms.Compose([
            # tv.transforms.Grayscale(),
            tv.transforms.Resize((img_size, img_size), interpolation= tv.transforms.InterpolationMode.BILINEAR, antialias=None)
        ])
        dataset = LungSegDataset(split="test", transforms=transforms)
        shuffle = True
    if split == 'val':
        transforms = tv.transforms.Compose([
            # tv.transforms.Grayscale(),
            tv.transforms.Resize((img_size, img_size), interpolation= tv.transforms.InterpolationMode.BILINEAR, antialias=None)
        ])
        dataset = LungSegDataset(split="val", transforms=transforms)
        shuffle = False
        
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_dataloaders)

if __name__ == "__main__":
   # dataset = LungSegDataset(split='train')
   # print(dataset.__getitem__(0))
   dataloader = make_dataloader(100, 0)
   item = next(iter(dataloader))
   #print(item)
   print(item["image"].size())