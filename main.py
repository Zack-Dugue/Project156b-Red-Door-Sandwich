import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import tensorboard
from XRAYdataLoader import make_dataloader
from model import XRAYModel
import time
import sys
import os
import argparse
NUM_CLASSES = 9


class XrayModule(LightningModule):
    def __init__(self,model,optimizer=None):
        super(XrayModule,self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.LossFun = nn.BCELoss(reduce=False)
    def forward(self,x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.model.train(False)

    def training_step(self,batch,batch_idx):
        x ,y  = batch
        y = y.to(th.float)
        (img,nan_mask) = x
        y_hat = self(img)
        # rn we still consider no finding. That should be removed from the training step at some point.
        losses = self.LossFun(y_hat[:,1:]*nan_mask[:,1:],y[:,1:])
        # pathologies
        losses = th.mean(losses,dim=0)
        loss = losses.mean()
        losses = losses

        tensorboard_logs = {'train_loss':loss, 'Enlarged Cardiomediastinum Loss' : losses[0],"Cardiomegaly" : losses[1],
                            "Lung Opacity" :losses[2],"Pneumonia" : losses[3],"Pleural Effusion" : losses[4],
                            "Pleural Other" : losses[5], "Fracture": losses[6], "Support Devices" : losses[7]}
        return {'loss':loss,'log':tensorboard_logs}

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.model.parameters(),lr=0.001)
        return self.optimizer
    def validation_step(self,batch,batch_idx):
        x, y= batch
        (img, nan_mask) = x
        self.train(False)
        y_hat = self(img)
        loss = self.LossFun(y_hat * nan_mask, y.to(th.float) )
        tensorboard_logs = {'validation_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        img, nan_mask = x
        return self(img)
    
    # def on_validation_epoch_end(self,outputs):
    #     avg_loss = th.stack([x['loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}
def experiment(path,model_name, num_nodes,num_dataloaders,batch_size,learning_rate,num_epochs):
    print("Using", th.device)
    
    # if th.device != 'cuda':
    #     accelerator = "cpu"
    accelerator = "auto"
    devices = "auto"
    if num_nodes== 1:
        strategy = "auto"
    else:
        strategy = pl.DDPStrategy(static_graph = False)
    trainer = pl.Trainer(accelerator = accelerator, devices=devices, max_epochs = num_epochs, strategy=strategy, num_nodes=num_nodes, log_every_n_steps=1)
    # ANNOTATIONS_LABELS = "C:\\Users\\dugue\\PycharmProjects\\Project156b-Red-Door-Sandwich\\data\\student_labels\\train_sample.csv"
    ANNOTATIONS_LABELS = os.path.join(os.getcwd(), 'data', 'student_labels', 'train2023.csv')
    train_loader = make_dataloader(ANNOTATIONS_LABELS, batch_size, num_dataloaders=num_dataloaders, train=True)
    # ANNOTATIONS_LABELS = "C:\\Users\\dugue\\PycharmProjects\\Project156b-Red-Door-Sandwich\\data\\student_labels\\train_sample.csv"
    ANNOTATIONS_LABELS = os.path.join(os.getcwd(), 'data', 'student_labels', 'train2023.csv')
    #For now training and validation are done on the same dataset
    validation_loader = make_dataloader(ANNOTATIONS_LABELS, batch_size, num_dataloaders=num_dataloaders, train=False)
    xray_model = XRAYModel(NUM_CLASSES)

    optimizer = th.optim.Adam(xray_model.parameters(),lr=learning_rate)
    print("Trainer system parameters:")
    print(f"\t trainer.world_size : {trainer.world_size}")
    print(f"\t trainer.num_nodes : {trainer.num_nodes}")
    print(f"\t trainer.accelerator : {trainer.accelerator}")
    print(f"\t trainer.device_ids {trainer.device_ids}")
    print(f"\t train_loader.num_workers : {train_loader.num_workers}")
    trainer.fit(XrayModule(xray_model,optimizer),train_loader,validation_loader)
    print("Training run complete")
    th.save(trainer.model.state_dict(), os.path.join(path, model_name + ".pth"))
    print("Model Saved, experiment complete.")


#The Arguments of the Command Line are the following:
# Path, Model_Name, Number of Nodes, Number of Dataloaders, Batch Size, Learning Rate, Number of Epochs

if __name__ == "__main__":
    
    print("Running Experiment: ")
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', help='Name of experiment folder', default='test_1', type=str)
    parser.add_argument('-N', '--name', help='Name of the model', default='MODEL_1', type=str)
    parser.add_argument('-n', '--num_nodes', help='Number of nodes being run on', default=1, type=int)
    parser.add_argument('-d', '--num_dataloaders', help='Number of dataloader workers', default=1, type=int)
    parser.add_argument('-b', '--batch_size', help='Batch size', default=512, type=int)
    parser.add_argument('-l', '--learning_rate', help='Learning rate of model', default=.001, type=float)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs', default=20, type=int)
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), 'experiments', args.path)
    model_name = args.name
    num_nodes = args.num_nodes
    num_dataloaders = args.num_dataloaders
    batch_size = args.batch_size
    lr = args.learning_rate
    NumEpochs = args.num_epochs

    print(f"Model Name: {model_name} \t num_nodes: {num_nodes} \t num_dataloaders: {num_dataloaders}"
          f"\n batch_size: {batch_size} \t learning_rate: {lr} \t num_epochs: {NumEpochs}")
    try:
        os.mkdir(path)
    except FileExistsError:
        if len(os.listdir(path)) == 0:
            pass
        else:
            pass
            # raise FileExistsError
    print(f"Experiment Info and Files stored in:{path}")
    experiment(path,model_name,num_nodes,num_dataloaders,batch_size,lr,NumEpochs)
