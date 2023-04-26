import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

#Use lightning datamap database manager.
class XrayDataset(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = self.img_labels["Path"]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        labelX = self.img_labels.iloc[idx].to_dict()
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(label["Path"])
        pid = label["Path"].split("/")[1]

        X = (pid,label["Sex"], label["Age"], label["Frontal/Lateral"],label["AP/PA"])

        Y = (label["No Finding"], label['Enlarged Cardiomediastinum'], label["Cardiomegaly"], label['Lung Opacity'],
            label['Pneumonia'], label['Pleural Effusion'], label['Pleural Other'], label['Fracture'], label['Support Devices'])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #Might have to reformat this:
        # IE Image,X might need to be ina  tuple or something
        return image, X, Y

def make_custom_dataloader(*args, train = True):
    """
    Returns a dataloader for our dataset
    :param args: whatever args you think are appropriate
    :return:
    """
    dataset = CustomImageDataset()

def train_image_transform(*args):
    """
    Returns a transfrom which performs image augmentations during training
    :param args: whatever args you think are appropriate
    :return:
    """
    transform = tv.transforms.Compose([
        #TODO:
    ])
    return transform

def validation_image_transform(*args):
    """
    Returns a transform specifically for images during validation
    :param args: whatever args you think are appropriate
    :return:
    """
    transform = tv.transforms.Compose([
        #TODO:
    ])
    return transform