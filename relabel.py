import torchxrayvision as xrv
import torch as th
import torch.nn as nn
import torch.nn.utils as F
import csv
from XRAYdataLoader import XrayDataset, make_dataloader


def relabel_dataset(dataset, model, size, write_path):
    resize = xrv.datasets.XRayResizer(size)

    with open(write_path,'x') as csv_file:
        csv.writer(csv_file)
        for i in range(len(dataset)):
            img = dataset[i]
            img = resize(img)
            path = dataset.get_path(i)
            y = model(img)[0:9]
            lst = [i, path]
            lst.extend(list(y))
            csv_file.writerow(lst.extend(list(y)))

ANNOTATION_PATH = ""
if __name__ == '__main__':
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic")
    dataset = XrayDataset(ANNOTATION_PATH)
    size = 224
    write_path = "/data/student_labels/relabeled.csv"
