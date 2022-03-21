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

METRIC_ORDER = ['acc', 'auroc', 'auprc', 'precision', 'sensitivity', 'specificity', 'npv', 'npvsc']

checkpoint_paths = [
    '/home/csestili/repos/mouse_sst/lightning_logs/version_2608479/checkpoints/epoch=79-step=137519.ckpt',
    '/home/csestili/repos/mouse_sst/lightning_logs/version_2608480/checkpoints/epoch=80-step=139238.ckpt',
    '/home/csestili/repos/mouse_sst/lightning_logs/version_2608636/checkpoints/epoch=75-step=130643.ckpt'
]

def eval(checkpoint_paths):
    for idx, path in enumerate(checkpoint_paths):
        with torch.no_grad():
            model = CNN.load_from_checkpoint(path)
            y_hat, targets = get_predictions(model)
            print(f"model {idx}, path {path}")
            print_metrics_model(model, y_hat, targets)
            print()
            del model, y_hat, targets
            torch.cuda.empty_cache()

def get_predictions(model, use_cache=False):
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

        if use_cache and os.path.exists(y_hat_path) and os.path.exists(target_path):
            y_hat = torch.load(y_hat_path)
            targets = torch.load(target_path)
        else:
            y_hat = []
            targets = []
            for xs, ys in dataloader:
                y_hat.append(model(xs.to(0)))
                targets.append(ys)
            y_hat = torch.cat(y_hat)
            targets = torch.cat(targets).to(0)

            if use_cache:
                torch.save(y_hat, y_hat_path)
                torch.save(targets, target_path)

        y_hat = y_hat.exp()[:,1]

    return y_hat, targets

def print_metrics_model(model, y_hat, targets):
    precision, recall, _ = model.pr_curve(y_hat, targets)
    auprc = model.auprc(recall, precision)
    precision, recall, _ = model.negative_pr_curve(1 - y_hat, 1 - targets)
    npvsc = model.npvsc(recall, precision)
    metrics = {}
    metrics.update(model.metrics(y_hat, targets))
    metrics.update(model.negative_metrics(1 - y_hat, 1 - targets))
    metrics.update({
        'auprc': auprc,
        'npvsc': npvsc})
    print(metrics)
    print(','.join(
        format(metrics[metric].detach().cpu().numpy(), '.4f')
        for metric in METRIC_ORDER))

def print_metrics_torchmetrics(y_hat, targets):
    auroc_metric = torchmetrics.AUROC()
    pr_curve = torchmetrics.PrecisionRecallCurve()
    auc = torchmetrics.AUC(reorder=True)
    auroc = auroc_metric(y_hat, targets)
    precision, recall, thresholds = pr_curve(y_hat, targets)
    auprc = auc(recall, precision)
    precision, recall, thresholds = pr_curve(1 - y_hat, 1 - targets)
    npvsc = auc(recall, precision)

    print((f"""
        torchmetrics metrics:
        AUROC: {auroc}
        AUPRC: {auprc}
        NPVSC: {npvsc}
        """).strip())

def print_metrics_sklearn(y_hat, targets):
    import sklearn.metrics
    targets = targets.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()

    auroc = sklearn.metrics.roc_auc_score(targets, y_hat, average=None)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(targets, y_hat)
    auprc = sklearn.metrics.auc(recall, precision)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(1 - targets, 1 - y_hat)
    npvsc = sklearn.metrics.auc(recall, precision)

    print((f"""
        sklearn metrics:
        AUROC: {auroc}
        AUPRC: {auprc}
        NPVSC: {npvsc}
        """).strip())

if __name__ == "__main__":
    eval(checkpoint_paths)