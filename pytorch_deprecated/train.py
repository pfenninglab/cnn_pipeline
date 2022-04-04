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


SHOW_MODEL_SUMMARY = False
ARCH_PARAMS = {
    "conv_filters": 300,
    "conv_width": 7,
    "dense_filters": 300,
    "max_pool_size": 26,
    "max_pool_stride": 26,
    "dropout_rates": [0.3, 0.2, 0.1],
    "batch_size_train": 128,
    "batch_size_val": 128,
    "base_lr": 1e-5,
    "l2_reg_weight": 1e-4,
    "num_epochs": 100
}


class CNN(LightningModule):
    def __init__(self):
        super().__init__()

        # transposed input
        # TODO this doesn't have to be hardcoded if we couple the dataset to the LightningModule
        expected_input_shape = (4, 500)

        self.conv1 = nn.Conv1d(4, ARCH_PARAMS['conv_filters'], ARCH_PARAMS['conv_width'])
        self.conv2 = nn.Conv1d(ARCH_PARAMS['conv_filters'], ARCH_PARAMS['conv_filters'], ARCH_PARAMS['conv_width'])

        self.maxpool1 = nn.MaxPool1d(
            ARCH_PARAMS['max_pool_size'], stride=ARCH_PARAMS['max_pool_stride'])

        self.flatten1 = nn.Flatten()

        dim = get_shape_from_layers(
            [self.conv1, self.conv2, self.maxpool1, self.flatten1], expected_input_shape)
        self.linear1 = nn.Linear(dim[0], ARCH_PARAMS['dense_filters'])
        self.linear2 = nn.Linear(ARCH_PARAMS['dense_filters'], 2)

        self.dropout_conv1 = nn.Dropout(p=ARCH_PARAMS['dropout_rates'][0])
        self.dropout_conv2 = nn.Dropout(p=ARCH_PARAMS['dropout_rates'][1])
        self.dropout_linear1 = nn.Dropout(p=ARCH_PARAMS['dropout_rates'][2])

        self.metrics = torchmetrics.MetricCollection(
            {'acc': torchmetrics.Accuracy(),
            'auroc': torchmetrics.AUROC(),
            'sensitivity': torchmetrics.Recall(),
            'specificity': torchmetrics.Specificity(),
            'precision': torchmetrics.Precision()})
        self.negative_metrics = torchmetrics.MetricCollection(
            {'npv': torchmetrics.Precision()})
        self.pr_curve = torchmetrics.PrecisionRecallCurve()
        self.auprc = torchmetrics.AUC(reorder=True)
        self.negative_pr_curve = torchmetrics.PrecisionRecallCurve()
        self.npvsc = torchmetrics.AUC(reorder=True)


    def forward(self, x):
        x = x.transpose(1, 2).float()
        x = self.dropout_conv1(F.relu(self.conv1(x)))
        x = self.dropout_conv2(F.relu(self.conv2(x)))
        x = self.maxpool1(x)
        x = self.flatten1(x)
        x = self.dropout_linear1(F.relu(self.linear1(x)))
        x = self.linear2(x)

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

        # y_hat is log-"probabilities", so exp(y_hat) is "probabilities"
        # column 1 is "probability" of positive class
        y_hat = y_hat.exp()[:,1]
        self.metrics.update(y_hat, y)
        self.log_dict(self.metrics, on_epoch=True)

        self.pr_curve.update(y_hat, y)

        # metrics where the negative class is considered positive
        self.negative_metrics.update(1 - y_hat, 1 - y)
        self.log_dict(self.negative_metrics, on_epoch=True)

        self.negative_pr_curve.update(1 - y_hat, 1 - y)

    def validation_epoch_end(self, validation_step_outputs):
        precision, recall, _ = self.pr_curve.compute()
        self.auprc.update(recall, precision)
        self.log("auprc", self.auprc, on_epoch=True)

        precision, recall, _ = self.negative_pr_curve.compute()
        self.npvsc.update(recall, precision)
        self.log("npvsc", self.npvsc, on_epoch=True)

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=ARCH_PARAMS['base_lr'],
            weight_decay=ARCH_PARAMS['l2_reg_weight'])

def get_shape_from_layers(layers, image_dim: tuple):
    shape = (1,) + image_dim
    for layer in layers:
        shape = get_output_shape(layer, shape)
    return shape[1:]

def get_output_shape(model, image_dim):
    """from https://stackoverflow.com/a/62197038"""
    return model(torch.rand(*(image_dim))).data.shape


if __name__ == '__main__':

    model = CNN()
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
        max_epochs=ARCH_PARAMS['num_epochs'])

    train_dataloader = DataLoader(dataset.FaDatasetSampler(part='train', random_skip_range=8),
        batch_size=ARCH_PARAMS['batch_size_train'])
    val_dataloader = DataLoader(dataset.SinglePassDataset(part='val'),
        batch_size=ARCH_PARAMS['batch_size_val'])

    trainer.validate(model=model, dataloaders=val_dataloader)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
