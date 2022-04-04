# Classifier training tool

## About
This tool enables you to train a CNN classifier on DNA sequences.

Specify datasets, labels, CNN architecture, hyperparameters and training details
from a text configuration file.

Track experiments, visualize performance metrics, and search hyperparameters using
the `wandb` framework.

## Setup
1. Create conda environment with `tensorflow==2.7.0`.

(TODO) automatically create environment from script or env yaml file.

2. Create `wandb` account and authenticate from `lane`.

## Usage
1. Edit `config-mouse-sst.yaml` to configure:
- data sources (FASTA files)
- labels
- CNN architecture
- regularization
- learning rate and optimizer
- training details

2. Start training:
```
sbatch train.sb config-mouse-sst.yaml
```

3. Check experiment results in-browser at [https://wandb.ai/](https://wandb.ai/).
