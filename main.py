import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as TV
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
import tensorboard
from XRAYdataLoader import CustomDataLoader
from model import XRAYModel

class XrayModule(LightningModule):
    def __init__(self,model,optimizer=None):
        super(XrayModule,self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.LossFun = nn.BCELoss()
    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x ,y  = batch
        self.model.eval()
        y_hat = self(x)[0]
        self.model.train(False)
        loss = self.LossFun(y_hat,y)
        #TODO include something to actually evaluate the accuracy
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss,'log':tensorboard_logs}

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = th.optim.Adam(self.model.parameters(),lr=0.001)
        return self.optimizer
    def validation_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.LossFun(y_hat, y)
        #TODO include something to actually evaluate the accuracy
        tensorboard_logs = {'validation_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self,outputs):
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
def experiment(num_nodes = 1, num_dataloaders = 1):
    accelerator = th.device
    if num_nodes== 1:
        strategy = "auto"
    else:
        strategy = pl.DDPStrategy(static_graph = False)
    trainer = pl.Trainer(accelerator = accelerator, strategy=strategy, num_nodes=num_nodes, num_dataloaders=num_dataloaders,logger=T)
    # TODO: Finish Dataloaders
    train_loader = make_dataloader(*params,train=True)
    validation_loader = make_dataloader(*params,train=False)
    trainer.fit(XRAYModel(),train_loader,validation_loader)

if __name__ == "__main__":
    experiment()