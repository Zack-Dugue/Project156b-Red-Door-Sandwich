import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


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
        return image, X, Y