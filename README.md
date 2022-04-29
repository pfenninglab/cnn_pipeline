# Classifier training tool

## About
This tool enables you to train a CNN classifier on DNA sequences.

Specify datasets, labels, CNN architecture, hyperparameters and training details
from a text configuration file.

Track experiments, visualize performance metrics, and search hyperparameters using
the `wandb` framework.

## Setup
1. Create conda environments `keras2-tf27` (for training) and `keras2-tf24` (for SHAP/TF-MoDISco interpretation):

```
sbatch setup.sb
```

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

### Single training run
To train a single model:

1. Edit `config-base.yaml` to configure:
- `wandb` project name for tracking
- data sources (FASTA files)
- labels
- CNN architecture
- regularization
- learning rate and optimizer
- training details
- SHAP details

 Save the new config as `<my-config>.yaml`.

2. Start training:
```
sbatch train.sb <my-config>.yaml
```

3. Check experiment results in-browser at [https://wandb.ai/](https://wandb.ai/).

Trained models are saved in the `wandb/` directory.

### Hyperparameter sweep
To initiate a hyperparameter sweep, training many models with different hyperparameters:

1. Edit `config-base.yaml` as above, for all the parameters that should remain *fixed* during training.

2. Edit `sweep-config.yaml`, specifying all the parameters that should vary during the search, as well as the ranges to search over. If you saved your copy of `config-base.yaml` under a different name in step 1, be sure to change the base config name in the `command` section of `sweep-config.yaml`.

3. Start the sweep:
```
srun -n 1 -p pool1 --pty ./start_sweep.sh sweep-config.yaml
```
This will output a sweep id, e.g. `csestili-cmu/test-sweep/kztk7ceb`. Make note of it for the next step.

4. Start the sweep agents in parallel:
```
sbatch --array=1-<num_agents>%<throttle> start_agents.sb <sweep_id>
```
where
- `<sweep_id>` is the sweep id you got in step 3
- `<num_agents>` is the total number of agents you want to run in the sweep
- `<throttle>` is the maximum number of agents to run simultaneously. Please use this to keep resources free for other users!

5. Check sweep results in-browser at [https://wandb.ai/](https://wandb.ai/).

Trained models are saved in the `wandb/` directory.
