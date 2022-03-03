import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary
import torchmetrics

import dataset


SHOW_MODEL_SUMMARY = True
ARCH_PARAMS = {
    "conv_filters": 300
}


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Conv1d(4, ARCH_PARAMS['conv_filters'], 3)
        self.layer_2 = nn.Conv1d(ARCH_PARAMS['conv_filters'], ARCH_PARAMS['conv_filters'], 3)
        self.layer_3 = nn.Conv1d(ARCH_PARAMS['conv_filters'], 2, 3)
        self.layer_4 = nn.Flatten()
        self.layer_5 = nn.Linear(2 * 494, 2)

        self.metrics = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(),
            torchmetrics.Precision(average=None, num_classes=2),
            torchmetrics.Recall(average=None, num_classes=2)])

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

        metrics = self.metrics(y_hat, y)
        self.log_dict(metrics, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-5)


if __name__ == '__main__':

    model = LitMNIST()
    if torch.cuda.is_available():
        model.cuda()

    if SHOW_MODEL_SUMMARY:
        for seq, _ in dataset.FaDataset('train'):
            dims = seq.shape
            break
        print(summary(model, dims))

    strategy = None
    if torch.cuda.device_count() > 1:
        strategy = 'ddp'

    # TODO try auto_lr_find automatic learning rate finder
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        strategy=strategy,
        precision=16,
        max_epochs=100)

    train_dataloader = DataLoader(dataset.FaDataset(part='train'), batch_size=512)
    val_dataloader = DataLoader(dataset.FaDataset(part='val'), batch_size=512)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
