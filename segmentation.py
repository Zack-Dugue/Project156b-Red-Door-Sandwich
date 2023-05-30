import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import tensorboard
import time
import sys
import os
import argparse
from torch.utils.data import DataLoader , Dataset
from segmentationDataLoader import make_dataloader
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from torchvision.utils import save_image


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015)
    https://arxiv.org/abs/1505.04597

    Contracting Path
        - Two 3x3 Conv2D (Unpadded Conv, i.e. no padding)
        - followed by a ReLU
        - A 2x2 MaxPooling (with stride 2)
    Expansive Path : sequence of "up-convolutions" and "concatenation" with high-resolution feature from contracting path
        - "2x2 up-convolution" that halves the number of feature channels
        - A "concatenation" with the correspondingly cropped feature map from the contracting path
        - Two 3x3 Conv2D
        - Followed by a ReLU

    Final Layer
        - "1x1 Conv2D" is used to map each 64 component feature vector to
        the desired number of classes
    """
    def __init__(self, n_channels, n_classes , bilinear = False):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.in_conv = UNetConvBlock(self.n_channels , 64)
        self.Down1 = Down(64 , 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 512)
        self.Up1 = Up(512 + 512, 256 , self.bilinear)
        self.Up2 = Up(256 + 256, 128 , self.bilinear)
        self.Up3 = Up(128 + 128 , 64 , self.bilinear)
        self.Up4 = Up(64 + 64, 64 , self.bilinear)
        self.out_conv = OutConv(64, n_classes)

    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)
        x = self.Up1(x5,x4)
        x = self.Up2(x ,x3)
        x = self.Up3(x ,x2)
        x = self.Up4(x ,x1)
        out = self.out_conv(x)
        return out


class Down(nn.Module):
    """
    Downscaling with maxpool and then Double Conv
        - 3x3 Conv2D -> BN -> ReLU
        - 3X3 Conv2D -> BN -> ReLU
        - MaxPooling
    """
    def __init__(self, in_channels , out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetConvBlock(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
        - Upsampling Convolution ("Up-convolution")
        - 3x3 Conv2D -> BN -> ReLU
        - 3X3 Conv2D -> BN -> ReLU

        Upsampling vs Transposed convolutions

        Transposed convolution (a.k.a."up-convolution or fractionally-strided convolutions or deconvolutions")
            - The original paper uses this
            - detects more fine-grained detail

        Other implementation use bilinear upsampling, possibly followed by a 1x1 convolution.
        The benefit of using upsampling is that it has no parameters and if you include the 1x1 convolution,
        it will still have less parameters than the transposed convolution
    """
    def __init__(self,in_channels , out_channels , bilinear = False):
        super(Up,self).__init__()

        if bilinear: # use the normal conv to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode= 'bilinear', align_corners = True)
        else: # use Transpose convolution (the one that official UNet used)
            self.up = nn.ConvTranspose2d(in_channels//2 , in_channels // 2, kernel_size = 2,stride=2 )

        self.conv = UNetConvBlock(in_channels,out_channels)

    def forward(self,x1,x2):
        # input dim is CHW
        x1 = self.up(x1)

        diffY = th.tensor([x2.size()[2] - x1.size()[2]])
        diffX = th.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1 , [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = th.cat([x2, x1] , dim = 1)
        out = self.conv(x)
        return out


class UNetConvBlock(nn.Module):
    " [conv -> BN -> ReLU] -> [conv -> BN -> ReLU]"
    def __init__(self, in_channels, out_channels, kernel_size = 3 , padding = True):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:

            The Original paper uses VALID padding (i.e. no padding)
            The main benefit of using SAME padding is that the output feature map will have the same spatial dimensions
            as the input feature map.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            # Usually Conv -> BatchNormalization -> Activation
            nn.Conv2d(in_channels , out_channels , kernel_size= kernel_size , padding = int(padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=int(padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self,inp):
        return self.double_conv(inp)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetModule(LightningModule):
    def __init__(self, model, optimizer=None):
        super(UNetModule, self).__init__()
        self.model = model
        self.optimizer = optimizer
        # if n_classes is > 1, use cross entropy loss
        self.LossFun = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self) -> None:
        self.model.train(True)
    
    def training_step(self, batch, batch_idx):
        img, mask = batch['image'], batch['mask']
        #print(img.size())
        pred_mask = self(img)
        loss = self.LossFun(pred_mask, mask)
        return loss
    
    def configure_optimizers(self):
        if not self.optimizer:
            self.optimizer = th.optim.Adam(model.parameters(), lr= args.lr, weight_decay=1e-5)
        return self.optimizer
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch['image'], batch['mask']
        pred_mask = self(img)
        loss = self.LossFun(pred_mask, mask)
        self.log("val_loss", loss)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lr', type=float, help='The learning rate', default=.05)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=5)
    parser.add_argument('-d', '--num_dataloaders', type=int, default=8)
    parser.add_argument('-m', '--model_state', type=str, default=None)

    args = parser.parse_args()

    model = UNet(n_channels=1, n_classes=1)
    optimizer = th.optim.Adam(model.parameters(), lr= args.lr, weight_decay=1e-5)
    module = UNetModule(model, optimizer=optimizer)
    
    if args.model_state:
        module.load_state_dict(th.load(args.model_state, map_location=th.device('cuda')))
    if args.epochs > 0:
        train_data = make_dataloader(args.batch_size, args.num_dataloaders, split='train')
        val_data = make_dataloader(args.batch_size, args.num_dataloaders, split='val')
        trainer = pl.Trainer(accelerator="cuda", devices=1, strategy="auto", max_epochs=args.epochs, callbacks=[EarlyStopping(monitor="val_loss", mode='min', patience=5)])
        trainer.fit(module, train_dataloaders=train_data, val_dataloaders=val_data)
        th.save(module.state_dict(), os.path.join(os.getcwd(), 'experiments', 'segmentation', 'segmenter.pth'))


    # give an example
    test_data = make_dataloader(batch_size=8, num_dataloaders=args.num_dataloaders, split='test')
    images = next(iter(test_data))["image"]
    masks = module(images).detach()
    bool_masks = th.empty(masks.size(), dtype=bool)
    output_images = []
    for idx, mask in enumerate(masks):
        bool_masks[idx] = th.where(mask > 1e7, True, False)
        output_images.append(th.where(bool_masks[idx], 255, images[idx]).numpy().transpose(1, 2, 0))

    # plt.figure()

    f, axarr = plt.subplots(2, 4)
    for i in range(4):
        for j in range(2):
            axarr[j][i].imshow(output_images[2*i + j])
    plt.show()
    # cv.imshow('image', output_images[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # for idx, th_img in enumerate(bool_masks):
    #     print(th.min(th_img.to(th.float)))
    #     save_image(th_img.to(th.float), os.path.join('images', str(idx)+'.png'))