import os
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset
import pandas as pd
import torchvision as tv
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
        label = self.img_labels.iloc[idx].to_dict()
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(label["Path"])
        image = None
        pid = label["Path"].split("/")[1]


        Y = (label["No Finding"], label['Enlarged Cardiomediastinum'], label["Cardiomegaly"], label['Lung Opacity'],
            label['Pneumonia'], label['Pleural Effusion'], label['Pleural Other'], label['Fracture'], label['Support Devices'])
        Y = th.tensor(Y, dtype=th.float32)
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
        return (image, nan_mask), Y

def make_custom_dataloader(*args, train = True):
    """
    Returns a dataloader for our dataset
    :param args: whatever args you think are appropriate
    :return:
    """
    dataset = CustomImageDataset()
    # return DataLoader(dataset, batch_size, shuffle=True )
    
def train_image_transform(crop_size, rot_deg_range, hflip_p):
    """
    Returns a transfrom which performs image augmentations during training
    :param args: 
        - crop_size: tuple of crop size
        - rot_deg_range: degree range to rotate the image
        - hflip_p: probability of doing a horizontal flip
    :return:
    """
    #TODO: To normalize the data,
    transform = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(size=crop_size), 
        tv.transforms.RandomRotation(degrees=rot_deg_range), 
        tv.transforms.RandomHorizontalFlip(p=hflip_p)
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


if __name__ == "__main__":
    annotations_filepath = os.path.join(os.getcwd(), 'data', 'student_labels', 'train2023.csv')
    data = XrayDataset(annotations_file=annotations_filepath)
    item = data.__getitem__(0)
    print(item)