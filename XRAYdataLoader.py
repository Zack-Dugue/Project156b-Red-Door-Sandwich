import os
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset
import pandas as pd
import torchvision as tv
from torchvision.io import read_image
from matplotlib import pyplot as plt
import cv2 as cv
import time

#Use lightning datamap database manager.
class XrayDataset(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None,train = True, chexpert = False, brax = False, mimic = False, alpha = 0):
        #get rid of annotations_file and use boolean parameters and csv filepath constants instead
        assert(chexpert or brax or mimic) #cant run the model without data
        label_list = []
        chexpertcsv = annotations_file #placeholders for respective csv filepaths
        braxcsv = None
        mimiccsv = None
        if chexpert:
            chex = pd.read_csv(chexpertcsv)
            label_list.append(chex)
        if brax:
            bra = pd.read_csv(braxcsv)
            label_list.append(bra)
        if mimic:
            mim = pd.read_csv(mimiccsv)
            label_list.append(mimiccsv)
        self.img_labels = pd.concat(label_list)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.alpha = alpha

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):


        label = self.img_labels.iloc[idx].to_dict()
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread("data/" + label["Path"],cv.IMREAD_GRAYSCALE)

        pid = label["Path"].split("/")[1]


        if self.train:
            Y = (label["No Finding"], label['Enlarged Cardiomediastinum'], label["Cardiomegaly"], label['Lung Opacity'],
                label['Pneumonia'], label['Pleural Effusion'], label['Pleural Other'], label['Fracture'], label['Support Devices'])
            Y = th.tensor(Y, dtype=th.float32)
            Y = th.where(Y == 0.0, .5 , Y)
            Y = th.where(Y == 1.0, 1.0-self.alpha, Y)
            Y = th.where(Y == -1.0, 0.0+self.alpha, Y)
        else:
            Y = th.tensor([0 for _ in range(9)])
        nans = th.isnan(Y)
        Y[nans] = 0
        nan_mask = th.logical_not(nans).to(th.float32)
        


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # Might have to reformat this:
        # IE Image,X might need to be ina  tuple or something
        # TODO: Include frontal vs Lateral or PA AP if in file name
        # print(f"idx - {idx} , image_shape -{image.size()}")
        return (image, nan_mask), Y
    def get_img(self,idx):

        label = self.img_labels.iloc[idx].to_dict()
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image("data/" + label["Path"])
        image = image.to(th.float32)/255.0
        return image

    def get_path(self,idx):
        label = self.img_labels.iloc[idx].to_dict()
        return "data/" + label["Path"]


def make_dataloader(annotations_file, batch_size, num_dataloaders, train = True):
    """
    Returns a dataloader for our dataset
    :param args: whatever args you think are appropriate
    :return:
    """
    if train == True:
        transform = train_image_transform(crop_size=(224, 224), rot_deg_range=10, hflip_p=0.5)
        shuffle = True
    else:
        transform = validation_image_transform((224,224))
        shuffle = False
    dataset = XrayDataset(annotations_file, transform=transform, target_transform=None, train = train, chexpert=True)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_dataloaders)

class CustomTransform(object):
    def __init__(self):
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.to_tensor = tv.transforms.ToTensor()
    def __call__(self, image):
        # image = cv.r
        image = self.clahe.apply(image)
        image = self.to_tensor(image)
        image = image.to(th.float32)/255.0
        return image


def train_image_transform(crop_size, rot_deg_range, hflip_p):
    """
    Returns a transfrom which performs image augmentations during training
    :param args: 
        - crop_size: tuple of crop size
        - rot_deg_range: degree range to rotate the image
        - hflip_p: probability of doing a horizontal flip
    :return:
    """

    transform = tv.transforms.Compose([
        # tv.transforms.Normalize( 0.533048452958796, 0.03490651403764978),
        CustomTransform(),

        tv.transforms.RandomResizedCrop(scale=(.85,1), ratio = (.9,1.1), interpolation= tv.transforms.InterpolationMode.BILINEAR , antialias=True, size=crop_size),

        tv.transforms.RandomRotation(degrees=rot_deg_range, interpolation=tv.transforms.InterpolationMode.BILINEAR)
        # tv.transforms.RandomHorizontalFlip(p=hflip_p)
    ])
    return transform

def validation_image_transform(size):
    """
    Returns a transform specifically for images during validation
    :param args: whatever args you think are appropriate
    :return:
    """
    transform = tv.transforms.Compose([
        # tv.transforms.Normalize( 0.533048452958796, 0.3490651403764978),
        CustomTransform(),
        tv.transforms.Resize(size, interpolation= tv.transforms.InterpolationMode.BILINEAR, antialias=True),

    ])
    return transform


####################################################################################################################
# For testing
####################################################################################################################



if __name__ == "__main__":
    # data = XrayDataset(annotations_file="data/student_labels/train_sample.csv", chexpert=True)
    data = make_dataloader("data/student_labels/train_sample.csv", 1,0)
    pneumonia = []
    no_pneumonia = []
    start = time.time()
    for i, item in enumerate(data):
        print(f"\rimage {i}",end="")
        image, nan_mask = item[0] # torch tensor of image and nan mask
        labels = item[1]
        # if labels[4] == 1:
        #     pneumonia.append(image.mean())
        # else:
        #     no_pneumonia.append(image.mean())
    finish = time.time()
    print(f"\ndone. It took {finish-start} seconds")
    print(sum(pneumonia)/len(pneumonia))
    print(sum(no_pneumonia)/len(no_pneumonia))
