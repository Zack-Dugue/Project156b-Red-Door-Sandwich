import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import tensorboard
from XRAYdataLoader import make_data_loader
from model import XRAYModel
import time
import sys
import os


class XrayModule(LightningModule):
    def __init__(self,model,optimizer=None):
        super(XrayModule,self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.LossFun = nn.BCELoss()
    def forward(self,x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.model.train(False)

    def training_step(self,batch,batch_idx):
        x ,y  = batch
        #Handle NAN Masking:
        y_hat = self(x)
        loss = self.LossFun(y_hat,F.one_hot(y,num_classes = 10).to(th.float32))
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss,'log':tensorboard_logs}

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.model.parameters(),lr=0.001)
        return self.optimizer
    def validation_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        self.model.train(False)
        loss = self.LossFun(y_hat, y)
        tensorboard_logs = {'validation_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # def on_validation_epoch_end(self,outputs):
    #     avg_loss = th.stack([x['loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}
def experiment(path,model_name, num_nodes,num_dataloaders,batch_size,learning_rate,num_epochs):
    if th.device != 'cuda':
        accelerator = "cpu"
    if num_nodes== 1:
        strategy = "auto"
    else:
        strategy = pl.DDPStrategy(static_graph = False)
    trainer = pl.Trainer(accelerator = accelerator,max_epochs = num_epochs, strategy=strategy, num_nodes=num_nodes)
    # TODO: Finish Dataloaders
    train_loader = make_dataloader(*params,train=True)
    validation_loader = make_dataloader(*params,train=False)
    xray_model = XRAYModel(10)

    optimizer = th.optim.Adam(xray_model.parameters(),lr=learning_rate)
    trainer.fit(XrayModule(xray_model,optimizer),train_loader,validation_loader)
    print("Training run complete")
    th.save(trainer.model.state_dict(), path + "\\" + model_name + ".pth")
    print("Model Saved, experiment complete.")


#The Arguments of the Command Line are the following:
# Path, Model_Name, Number of Nodes, Number of Dataloaders, Batch Size, Learning Rate, Number of Epochs

if __name__ == "__main__":
    print("Running Experiment: ")
    if len(sys.argv) <= 1:
        path = os.getcwd() + "\\experiments\\MNIST_TEST"
        model_name = "MNIST_MODEL"
        num_nodes = 1
        num_dataloaders = 1
        batch_size = 32
        lr = .001
        NumEpochs = 10
    else:
        args = sys.argv[1]
        path, model_name, num_nodes, num_dataloaders, batch_size, lr, NumEpochs = args[1:]
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
