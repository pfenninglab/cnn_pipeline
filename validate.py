"""validate.py: Validate a trained model.
This is the only way I've found to make the AUROC metric work as intended
(need to feed in predictions for entire validation dataset)
"""
from itertools import islice
import os

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch
import torchmetrics


from dataset import SinglePassDataset
from train import CNN

checkpoint_path = '/home/csestili/repos/mouse_sst/lightning_logs/base_arch/version_2604300/checkpoints/epoch=70-step=209946.ckpt'

model = CNN.load_from_checkpoint(checkpoint_path)

use_toy_example = False
if use_toy_example:
    # FP: 1
    # TP: 2
    # FN: 3
    # TN: 1
    targets = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    y_hat = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    y_hat = y_hat[:,1]
else:

    model.eval().to(0)
    dataset = SinglePassDataset(part='val', endless=False)
    dataloader = DataLoader(dataset, batch_size=1024)

    y_hat_path = '/home/csestili/repos/mouse_sst/tmp_y_hat.pt'
    target_path = '/home/csestili/repos/mouse_sst/tmp_target.pt'

    if not (os.path.exists(y_hat_path) and os.path.exists(target_path)):
        y_hat = []
        targets = []
        for xs, ys in dataloader:
            y_hat.append(model(xs.to(0)))
            targets.append(ys)
        y_hat = torch.cat(y_hat)
        targets = torch.cat(targets).to(0)

        torch.save(y_hat, y_hat_path)
        torch.save(targets, target_path)

    y_hat = torch.load(y_hat_path)
    targets = torch.load(target_path)

    y_hat = y_hat.exp()[:,1]



print(model.metrics(y_hat, targets))
print(model.negative_metrics(1 - y_hat, 1 - targets))



pr_curve = torchmetrics.PrecisionRecallCurve()
auc = torchmetrics.AUC(reorder=True)
precision, recall, thresholds = pr_curve(y_hat, targets)
print("AUPRC:")
print(auc(recall, precision))
print("NPVSC:")
precision, recall, thresholds = pr_curve(1 - y_hat, 1 - targets)
print(auc(recall, precision))

print("check:")
import sklearn.metrics
targets = targets.detach().cpu().numpy()
y_hat = y_hat.detach().cpu().numpy()

print("AUROC:")
print(sklearn.metrics.roc_auc_score(targets, y_hat, average=None))
print("AUPRC:")
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(targets, y_hat)
print(sklearn.metrics.auc(recall, precision))
print("NPVSC:")
precision, recall, _ = sklearn.metrics.precision_recall_curve(1 - targets, 1 - y_hat)
print(sklearn.metrics.auc(recall, precision))