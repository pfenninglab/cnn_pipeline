"""validate.py: Validate a trained model.
This is the only way I've found to make the AUROC metric work as intended
(need to feed in predictions for entire validation dataset)
"""
from itertools import islice

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch

from dataset import SinglePassDataset
from train import CNN

checkpoint_path = '/home/csestili/repos/mouse_sst/lightning_logs/base_arch/version_2604300/checkpoints/epoch=70-step=209946.ckpt'

model = CNN.load_from_checkpoint(checkpoint_path)
model.eval().to(0)
dataset = SinglePassDataset(part='val', endless=False)
dataloader = DataLoader(dataset, batch_size=1024)

y_hat = []
targets = []
for xs, ys in dataloader:
    y_hat.append(model(xs.to(0)))
    targets.append(ys)
y_hat = torch.cat(y_hat)
targets = torch.cat(targets).to(0)

print(model.post_train_metrics(y_hat, targets))
