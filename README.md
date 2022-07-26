# CNN Training Platform

## About
This tool enables you to train a CNN model on DNA sequences.
Classification (2 or more classes) and regression (single variable) are both supported.

Specify datasets, targets, CNN architecture, hyperparameters, and training details
from a text configuration file.

Track experiments, visualize performance metrics, and search hyperparameters using
the `wandb` framework.

## Cloning this repo
It is recommended that you use the SSH authentication method to clone this repo.

1. Start an interactive session:

```
srun -n 1 -p interactive --pty bash
```
On `bridges`, use `-p RM-shared` instead.

2. Create an SSH key and add it to your GitHub account, if you don't already have one:
[Instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys). You only need to do the 3 steps "Check for existing SSH key"
through "Add a new SSH key".

3. Clone the repo. It is recommended that you clone into a directory just for repositories:

```
mkdir ~/repos
cd ~/repos
git clone git@github.com:pfenninglab/mouse_sst.git
```

## Setup
1. Install Miniconda:
    1. Download the latest Miniconda installer. This is the correct installer for `lane` and `bridges`:
    
    ```
    cd /tmp
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
    
    If you're not on `lane` or `bridges`, check your system's architecture and download the correct
    installer from [Latest Miniconda Installer Links](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links).

    2. Run the installer:
    
    ```
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    
    You will need to answer a few questions when prompted. The default values for each question should work.

    3. Cleanup and go back to this repo:
    ```
    rm Miniconda3-latest-Linux-x86_64.sh
    cd ~/repos/mouse_sst
    ```

2. Create conda environments:

```
bash setup.sh <cluster_name>
```
where `<cluster_name>` is `lane` or `bridges`.

This creates the environments `keras2-tf27` (for training) and `keras2-tf24` (for SHAP/TF-MoDISco interpretation). This should take about 20 minutes.

3. Create a `wandb` account: [signup link](https://app.wandb.ai/login?signup=true)
 
 NOTE: `wandb` account usernames cannot be changed. I recommend creating a username like
 `<name>-cmu`, e.g. `csestili-cmu`, in case you want to have different accounts for personal
 use or for other future workplaces.

During account creation, you will be asked if you want to create a team. You do not need to do this.

4. Log in to `wandb` on `lane`:
```
srun -n 1 -p interactive --pty bash
conda activate keras2-tf27
wandb login
```
On `bridges`, use `-p RM-shared` instead.

## Model training

### Single training run
To train a single model:

1. Edit `config-base.yaml` to configure:
- `wandb` project name for tracking
- data sources
- targets
- CNN architecture
- regularization
- learning rate and optimizer
- training details
- SHAP details

For example training config files, see `config-classification.yaml` and `config-regression.yaml` in the `example_configs/` directory.

2. Start training:
```
bash train.sh config-base.yaml
```

3. Check experiment results in-browser at [https://wandb.ai/](https://wandb.ai/).

Trained models are saved in the `wandb/` directory.

### Hyperparameter sweep
To initiate a hyperparameter sweep, training many models with different hyperparameters:

1. Edit `config-base.yaml` as above, for all the parameters that should remain **fixed** during training.

2. Edit `sweep-config.yaml`, specifying all the parameters that should **vary** during the search, as well as the ranges to search over.

If you saved your copy of `config-base.yaml` under a different name in step 1, be sure to change the base config name in the `command` section of `sweep-config.yaml`.

3. Start the sweep:
```
bash start_sweep.sh sweep-config.yaml
```
This will output a sweep id, e.g. `<your wandb id>/<project name>/kztk7ceb`. Copy it for the next step.

4. Start the sweep agents in parallel:
```
bash start_agents.sh <num_agents> <throttle> <sweep_id>
```
where
- `<num_agents>` is the total number of agents you want to run in the sweep.
- `<throttle>` is the maximum number of agents to run simultaneously. It is recommended to set this to `4` or less. Please use this to keep resources free for other users!
- `<sweep_id>` is the sweep id you got in step 3.

5. Check sweep results in-browser at [https://wandb.ai/](https://wandb.ai/).

Trained models are saved in the `wandb/` directory.

## Using a trained model

### Finding a trained model

When you train a model, the training run gets an associated run ID.
The trained model is saved in a directory called `wandb/run-<date_string>-<run_id>`.

To find the directory associated with a given run:
1. Go to that run in the `wandb` user interface, e.g. https://wandb.ai/cmu-cbd-pfenninglab/mouse-sst/runs/1gimqghi .
2. The run ID is the part of that URL after `runs/`. E.g. for the above model, the run id is `1gimqghi`.
3. On `lane`, find the trained model with this run id. You can use the `find` command for this:
```
find wandb/ -wholename *<run id>*/files/model-final.h5
```

Note: The asterisks are part of the command. E.g. for the above model, you would use 
`find wandb/ -wholename *1gimqghi*/files/model-final.h5`.

This will give you a path to the trained model.

To get the final model, use `model-final.h5`.
To get the model with the lowest validation loss, use `model-best.h5`.

### Evaluating a trained model

To evaluate a trained model on one or more validation sets:

1. Edit `config-base.yaml` to include the paths to your datasets in  `additional_val_data_paths`, and the targets in `additional_val_targets`.
2. Run evaluation on your datasets:
```
cd mouse_sst/ (this repo)
srun -p pfen3 -n 1 --gres gpu:1 --pty bash
conda activate keras2-tf27
python -m scripts.validate -config config-base.yaml -model <path to model .h5 file>
```
This prints validation set metrics directly to your console.
To export the results to a .csv file, you can also use the flag `-csv <path to output .csv file>`.

**NOTE:** You can pass multiple validation datasets in to `additional_val_data_paths`. Each validation dataset can have 1 or more correct ground truth labels. Metrics for each dataset are reported separately. This is useful when some of your datasets have only positive examples, some have only negative examples, and some have a mixture of positive and negative examples. E.g.

```
additional_val_data_paths:
  value:
    - [positive_set_A.fa]
    - [negative_set_B.fa]
    - [negative_set_C.fa, positive_set_C.fa]

additional_val_targets:
  value:
    - [1]
    - [0]
    - [0, 1]
```

## Preprocessing
Currently, models expect all input sequences to be the same length, and will fail with a `ValueError`
if this is not the case.

If you have a `.bed` or `.narrowPeak` file with intervals of different lengths, you can use
`preprocessing.py` to produce another file with intervals of standardized lengths. Specifically,
the following preprocessing is applied:

**Input:** `length`, the desired standardized length.
1. Duplicate intervals are removed.
2. Each interval is replaced with another interval that has the same summit center, but is `length` bases long.

Usage:
```
python preprocessing.py expand_peaks -b <input .bed file> -o <output .bed file> -l <integer length>
```