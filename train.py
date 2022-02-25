import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary

import dataset


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Conv1d(4, 128, 3)
        self.layer_2 = nn.Conv1d(128, 128, 3)
        self.layer_3 = nn.Conv1d(128, 2, 3)
        self.layer_4 = nn.Flatten()
        self.layer_5 = nn.Linear(2 * 494, 2)

    def forward(self, x):
        x = x.transpose(1, 2).float()
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.relu(x)

        x = self.layer_4(x)
        x = self.layer_5(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


model = LitMNIST()
if torch.cuda.is_available():
    model.cuda()
print(summary(model, (500, 4)))
strategy = None
if torch.cuda.device_count() > 1:
    strategy = 'ddp'
trainer = Trainer(gpus=torch.cuda.device_count(), strategy=strategy)
data_loader = DataLoader(dataset.FaDataset('train'), batch_size=32)
trainer.fit(model, data_loader)