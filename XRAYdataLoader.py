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

    def __init__(self, annotations_file, transform=None, target_transform=None,train = True, chexpert = False, brax = False, mimic = False):
        #can we get rid of annotations_file and use boolean parameters and predifined csv filepaths instead?
        assert(chexpert or brax or mimic) #cant run the model without data
        label_list = []
        chexpertcsv = None #placeholders for respective csv filepaths
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

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):


        label = self.img_labels.iloc[idx].to_dict()
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image("data/" + label["Path"])
        image = image.to(th.float32)/255.0
        pid = label["Path"].split("/")[1]

        if self.train:
            Y = (label["No Finding"], label['Enlarged Cardiomediastinum'], label["Cardiomegaly"], label['Lung Opacity'],
                label['Pneumonia'], label['Pleural Effusion'], label['Pleural Other'], label['Fracture'], label['Support Devices'])
            Y = th.tensor(Y, dtype=th.float32)
            Y = th.where(Y == 0.0, .5 , Y)
            Y = th.where(Y == 1.0, 1.0, Y)
            Y = th.where(Y == -1.0, 0.0, Y)
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
        return (image.to(th.float32), nan_mask), Y

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
    dataset = XrayDataset(annotations_file, transform=transform, target_transform=None, train = train)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_dataloaders)
    
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
        tv.transforms.RandomResizedCrop(scale=(.8,1), interpolation= tv.transforms.InterpolationMode.BILINEAR , antialias=True, size=crop_size),
        tv.transforms.RandomRotation(degrees=rot_deg_range, interpolation=tv.transforms.InterpolationMode.BILINEAR),
        tv.transforms.RandomHorizontalFlip(p=hflip_p)
    ])
    return transform

def validation_image_transform(size):
    """
    Returns a transform specifically for images during validation
    :param args: whatever args you think are appropriate
    :return:
    """
    transform = tv.transforms.Compose([
        tv.transforms.Resize(size, interpolation= tv.transforms.InterpolationMode.BILINEAR, antialias=True),

    ])
    return transform


if __name__ == "__main__":
    data = XrayDataset(annotations_file="SampleLabels.csv")
    item = data.__getitem__(0)
    print(item)