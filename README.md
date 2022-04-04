# Classifier training tool

## About
This tool enables you to train a CNN classifier on DNA sequences.

Specify datasets, labels, CNN architecture, hyperparameters and training details
from a text configuration file.

Track experiments, visualize performance metrics, and search hyperparameters using
the `wandb` framework.

## Setup
1. Create conda environment `keras2-tf27`:

```
sbatch setup.sb
```
 This should take about 3-5 minutes to complete.

2. Create `wandb` account: [signup link](https://app.wandb.ai/login?signup=true)

 NOTE: `wandb` account usernames cannot be changed. I recommend creating a username like
 `<name>-cmu`, e.g. `csestili-cmu`, in case you want to have different accounts for personal
 use or for other future workplaces.

3. Log in to `wandb` on `lane`:
```
srun -n 1 -p interactive --pty bash
conda activate keras2-tf27
wandb login
```

## Usage
1. Edit `config-mouse-sst.yaml` to configure:
- `wandb` project name for tracking
- data sources (FASTA files)
- labels
- CNN architecture
- regularization
- learning rate and optimizer
- training details

 Save the new config as `<my-config>.yaml`.

2. Start training:
```
sbatch train.sb <my-config>.yaml
```

3. Check experiment results in-browser at [https://wandb.ai/](https://wandb.ai/).
