import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

class XRAYModel(LightningModule):
    def __init__(self,num_classes):
        super().__init__()
        self.base_model = tv.models.convnext_base().features
        # self.base_model = th.load("dino_vitbase8_pretrain.pth")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024,2048)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048,2048)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048,num_classes)
        self.act3 = nn.Sigmoid()
        self.eval = False
        print("Finished Initializing xRay Model")

    def parameters(self,freeze_features=True):
        if freeze_features:
            return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters()) + list(self.drop1.parameters()) + list(self.drop2.parameters()) + list(self.drop3.parameters())
        else:
            return list(self.base_model.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters()) + list(self.drop1.parameters()) + list(self.drop2.parameters()) + list(self.drop3.parameters())
    def forward(self,x : th.Tensor):
        x = self.base_model(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.act3(x)
        # TODO handle the whole "no finding" thing.

        return x


if __name__ =="__main__":
    model = XRAYModel(11)