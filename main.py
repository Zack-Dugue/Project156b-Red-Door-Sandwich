import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
import tensorboard
from XRAYdataLoader import make_dataloader
from model import XRAYModel
import time
import sys
import os
import argparse
NUM_CLASSES = 9
# Variance / Variance = 1
# MSE / Variance
class Criterion(nn.Module):
    def __init__(self, device,alpha = th.Tensor([0.0,0.05,0.05, 0.0,0.1,0.0, 0.15,0.18,0.0])):
        super().__init__()
        self.LossFun = nn.MSELoss(reduction = 'none')
        self.device = device
        var = th.Tensor([0.637237519,0.765471336,0.869690439,0.494558767,0.46360988,0.86843845,0.541910475,0.863346014,0.435933443])**2
        non_nan_frac = th.Tensor([0.999955096,0.217191675,0.229618986,0.531915849,0.122942814,0.598969443,0.030518198,0.059767844,0.564112351])
        self.weighting = var*non_nan_frac
        self.weighting = self.weighting.to(device)
        self.alpha = alpha.to(device)
    def adjust(self,y):
        y_ones_like = th.ones_like(y).to(self.device)
        y = y.to(self.device)*2 - y_ones_like
        y = y * (y_ones_like - self.alpha)
        y = (y.to(self.device) + y_ones_like)/2
        return y.to(self.device)
    def forward(self,y_hat,y : th.Tensor,nan_mask):
        y = self.adjust(y)
        unscaled_loss = self.LossFun(y_hat.to(self.device),y.to(self.device))
        # coaunscaled_loss = th.mean(unscaled_loss,dim=0).to(self.device)
        loss = unscaled_loss/self.weighting
        loss = loss * nan_mask.to(self.device)
        loss = th.mean(loss, dim=0).to(self.device)
        return loss

class XrayModule(LightningModule):
    def __init__(self,model,optimizer=None):
        super(XrayModule,self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.LossFun = Criterion(self.device)
    def forward(self,x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.model.train(True)

    def training_step(self,batch,batch_idx):
        self.train(True)
        x ,y  = batch
        # y = y.to(th.float)
        (img,nan_mask) = x
        y_hat = self(img)
        losses = self.LossFun(y_hat, y,nan_mask)
        # print(losses.shape)

        # pathologies
        loss = losses.mean()
        losses = losses
        self.logger.experiment.add_scalars('Losses', 
                                           {'train_loss': loss,
                                            'No Finding' : losses[0],
                                            'Enlarged Cardiomediastinum':losses[1],
                                            'Cardiomegaly':losses[2],
                                            'Lung Opacity':losses[3],
                                            'Pneumonia':losses[4],
                                            'Pleural Effusion':losses[5],
                                            'Pleural Other':losses[6],
                                            'Fracture':losses[7],
                                            'Support Devices':losses[8]},
                                            self.current_epoch)
        tensorboard_logs = {'train_loss':loss, 'No Finding' : losses[0],'Enlarged Cardiomediastinum' : losses[1],"Cardiomegaly" : losses[2],
                            "Lung Opacity" :losses[3],"Pneumonia" : losses[4],"Pleural Effusion" : losses[5],
                            "Pleural Other" : losses[6], "Fracture": losses[7], "Support Devices" : losses[8]}
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
        loss = self.LossFun(y_hat, y,nan_mask)
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
def experiment(path,model_name, num_nodes,num_dataloaders,batch_size,learning_rate,num_epochs, gpus):

    # accelerator = "cuda"
    accelerator = "auto"
    devices = gpus
    # devices = "auto"
    strategy = pl.strategies.DDPStrategy(static_graph = True)
    # strategy = "auto"
    # profiler = PyTorchProfiler(dirpath=path, filename='perf-logs')
    profiler = None
    logger = TensorBoardLogger(os.path.join(path, 'tb_logs'), name=model_name)
    trainer = pl.Trainer(accelerator = accelerator, devices=devices, max_epochs = num_epochs, strategy=strategy, 
                         num_nodes=num_nodes, log_every_n_steps=50, default_root_dir=path, profiler=profiler,
                         logger=logger)
    # ANNOTATIONS_LABELS = "C:\\Users\\dugue\\PycharmProjects\\Project156b-Red-Door-Sandwich\\data\\student_labels\\train_sample.csv"
    ANNOTATIONS_LABELS = os.path.join(os.getcwd(), 'data', 'student_labels', 'train2023.csv')
    train_loader = make_dataloader(ANNOTATIONS_LABELS, batch_size, num_dataloaders=num_dataloaders, train=True)
    # ANNOTATIONS_LABELS = "C:\\Users\\dugue\\PycharmProjects\\Project156b-Red-Door-Sandwich\\data\\student_labels\\train_sample.csv"
    # ANNOTATIONS_LABELS = os.path.join(os.getcwd(), 'data', 'student_labels', 'train2023.csv')
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
    parser.add_argument('-e', '--num_epochs', help='Number of epochs', default=2, type=int)
    parser.add_argument('-g', '--num_gpus', help='Number of gpus per node', default=4, type=int)
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), 'experiments', args.path)
    model_name = args.name
    num_nodes = args.num_nodes
    num_dataloaders = args.num_dataloaders
    batch_size = args.batch_size
    lr = args.learning_rate
    NumEpochs = args.num_epochs
    gpus = args.num_gpus

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
    #model_name = "fred"
    experiment(path,model_name,num_nodes,num_dataloaders,batch_size,lr,NumEpochs, gpus)
