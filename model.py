import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
class MSE_Class_Act(nn.Module):
    def __init__(self):
        """
        The goal here is to create an activation that squashes the output values between 0 and 1, but
        is compatible with the MSE objective. I think this does that better than sigmoid.
        Cause it has a low derivative when the output is near 1, when its near -1, and when its near 0.
        :param num_classes:
        """
        super().__init__()
        # self.a = nn.Parameter(th.ones_like(num_classes,requires_grad=True))4
        self.softsign = nn.Softsign()
    def forward(self,x):
        ones =  th.ones_like(x)
        # a = th.repeat(self.a,[x.size(0),1])
        return (self.softsign(x) + ones)/2

class MSE_Class_Act2(nn.Module):
    def __init__(self,num_classes):
        """
        Same as MSE one, however, we parameterize a bias towards zero using a value a.

        """
        super().__init__()
        self.a = nn.Parameter(th.ones_like(num_classes,requires_grad=True))
    def forward(self,x):
        ones =  th.ones_like(x)
        a = -th.repeat(self.a,[x.size(0),1])**2
        out = ones / (ones + th.abs(x)**(a)) * th.sign(x)
        return out


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(192)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        # x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)

        return x


class XRAYModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        print("Initializing XRAY Model")
        # self.base_model = tv.models.convnext_base().features
        # self.base_model = th.load("dino_vitbase16_pretrain.pth")
        # self.base_model = th.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.base_model = th.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        # self.early_norm = nn.BatchNorm2d(3,affine=False)
        print("Base Model Loaded")
        # self.base_model = TestModel()
        self.vit = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, 3072)
        # self.fc1 = nn.Linear(384,2048)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(3072,3072)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(3072,num_classes)
        self.act3 = MSE_Class_Act()
        self.eval = False
        print("Finished Initializing xRay Model")

    def parameters(self,freeze_features=True):
        if freeze_features:
            for param in self.base_model.parameters():
                param.requires_grad = False
            return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters()) + list(self.drop1.parameters()) + list(self.drop2.parameters())
        else:
            return list(self.base_model.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters()) + list(self.drop1.parameters()) + list(self.drop2.parameters())
    def forward(self,x : th.Tensor):
        x = x.repeat([1,3,1,1])
        # x = self.early_norm(x)

        x = self.base_model(x)
        if not self.vit:
            # x = self.avg_pool(x)
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