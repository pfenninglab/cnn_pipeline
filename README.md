# CNN Pipeline: Training, Evaluation, Inference, Interpretation
[Pfenning Lab](https://www.pfenninglab.org)

![CNN model illustration](https://github.com/pfenninglab/cnn_pipeline/blob/main/cnn%20model%20illustration.svg)

## About
This tool enables you to train a CNN model on DNA sequences.
Classification (2 or more classes) and regression (single variable) are both supported.

Specify datasets, targets, CNN architecture, hyperparameters, and training details
from a text configuration file.

Track experiments, visualize performance metrics, and search hyperparameters using
the `wandb` framework.

## Publications
### Context-dependent regulatory variants in Alzheimer’s disease
Ziheng Chen, Yaxuan Liu, Ashley R. Brown, Heather H. Sestili, Easwaran Ramamurthy, Xushen Xiong, Dmitry Prokopenko, BaDoi N. Phan, Lahari Gadey, Peinan Hu, Li-HueiTsai, Lars Bertram, Winston Hide, Rudolph E. Tanzi, Manolis Kellis, Andreas R. Pfenning  
bioRxiv 2025.07.11.659973; doi: [https://doi.org/10.1101/2025.07.11.659973](https://doi.org/10.1101/2025.07.11.659973)  
[https://www.biorxiv.org/content/10.1101/2025.07.11.659973v2.abstract](https://www.biorxiv.org/content/10.1101/2025.07.11.659973v2.abstract)

### Combining Machine Learning and Multiplexed, In Situ Profiling to Engineer Cell Type and Behavioral Specificity
Michael J. Leone, Robert van de Weerd, Ashley R. Brown, Myung-Chul Noh, BaDoi N.Phan, Andrew Z. Wang, Kelly A. Corrigan, Deepika Yeramosu, Heather H. Sestili, Cynthia M. Arokiaraj, Bettega C. Lopes, Vijay Kiran Cherupally, Daryl Fields, SudhagarBabu, Chaitanya Srinivasan, Riya Podder, Lahari Gadey, Daniel Headrick, Ziheng Chen, Michael E. Franusich, Richard Dum, David A. Lewis, Hansruedi Mathys, William R.Stauffer, Rebecca P. Seal, Andreas R. Pfenning  
bioRxiv 2025.06.20.660790; doi: [https://doi.org/10.1101/2025.06.20.660790](https://doi.org/10.1101/2025.06.20.660790)  
[https://www.biorxiv.org/content/10.1101/2025.06.20.660790v1.abstract](https://www.biorxiv.org/content/10.1101/2025.06.20.660790v1.abstract)

## Cloning this repo
It is recommended that you use the SSH authentication method to clone this repo.

1. Start an interactive session:

```
srun -n 1 -p interactive --pty bash
```
On `bridges`, use `-p RM-shared` instead.

2. Create an SSH key and add it to your GitHub account, if you don't already have one:
[Instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys). If you're on the Lane cluster or the PSC, be sure to select the **Linux** tab to see the correct instructions. You only need to do the 3 steps "Check for existing SSH key"
through "Add a new SSH key".

3. Clone the repo. It is recommended that you clone into a directory just for repositories:

```
mkdir ~/repos
cd ~/repos
git clone git@github.com:pfenninglab/cnn_pipeline.git
```

## Setup
1. Install `conda`, if it's not already installed:
    1. Check whether conda is installed:
    ```
    conda
    # if conda is already installed, you'll see:
    usage: conda [-h] [--no-plugins] [-V] COMMAND ...
    
    conda is a tool for managing and deploying applications, environments and packages.
    ```
    If conda is installed, then skip to step 2, **Create conda environments**. If conda is not installed,
    then follow these steps to install:
   
    2. Download the latest Miniconda installer. This is the correct installer for `lane` and `bridges`:
    
    ```
    cd /tmp
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
    
    If you're not on `lane` or `bridges`, check your system's architecture and download the correct
    installer from [Latest Miniconda Installer Links](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links).

    3. Run the installer:
    
    ```
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    
    You will need to answer a few questions when prompted. The default values for each question should work.

    4. Cleanup and exit the interactive node:
    ```
    rm Miniconda3-latest-Linux-x86_64.sh
    exit
    ```
    This will return you to the head node on the cluster.

3. Create conda environments:

Ensure that you are on the head node. Go to this repo and run the setup script:
```
cd ~/repos/cnn_pipeline
bash setup.sh <cluster_name>
```
where `<cluster_name>` is `lane` or `bridges`.

This creates the environments `keras2-tf27` (for training) and `keras2-tf24` (for SHAP/TF-MoDISco interpretation). This should take about 20 minutes.

3. Create a `wandb` account: [signup link](https://app.wandb.ai/login?signup=true)
 
 NOTE: `wandb` account usernames cannot be changed. I recommend creating a username like
 `<name>-cmu`, e.g. `hharper-cmu`, in case you want to have different accounts for personal
 use or for other future workplaces.

During account creation, you will be asked if you want to create a team.
You do not need to do this, so skip it if you're able to.
If wandb doesn't let you skip, then create a team named after your username, or any team name you prefer.

4. Log in to `wandb` on `lane`:
```
srun -n 1 -p interactive --pty bash
conda activate keras2-tf27
wandb login
```
On `bridges`, use `-p RM-shared` instead.

Once you have logged in to `wandb`, you can leave the interactive session and return to the head node:
```
exit
```

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

### Finding learning rate for Cyclic LR:
1. Fill out `config-base.yaml` with your train & validation data paths and model architecture.

2. Run the CLR learning rate range test (takes about 30 minutes on default dataset):
```
sbatch -n 1 -p pfen3 --gres gpu:1 --wrap "\
source activate keras2-tf27; \
python clr_rangetest.py -config config-base.yaml"
```
Parameters:
- `-config`: CNN pipeline config yaml file, e.g. config-base.yaml
- `-minlr`: Minimum LR in the search. Default `1e-6`.
- `-maxlr`: Maximum LR in the search. Default `50`.

3. The output, `lr_find/lr_loss.png`, is a plot of loss vs learning rate.
Look at the plot and use this to interpret it:
https://github.com/titu1994/keras-one-cycle/tree/master#interpreting-the-plot

4. The bounds you should use for the cyclic LR are:
- `lr_max`: the number you get from interpreting the plot, e.g. `10^(-1.7)`
- `lr_init`: lr_max / 20, e.g. `5^(-2.7)`
You might need to try other values close to these values.

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
cd cnn_pipeline/ (this repo)
srun -p pfen3 -n 1 --gres gpu:1 --pty bash
conda activate keras2-tf27
python scripts/validate.py -config config-base.yaml -model <path to model .h5 file>
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

### Get activations from a trained model

Get the outputs of a trained model, or the inner-layer activations, using `scripts/get_activations.py`:
```
Usage: python scripts/get_activations.py \
  -model <path to model .h5>
  -in_files <paths to input .fa, .bed, or .narrowPeak file> \
  [-in_genomes <paths to genome .fa file, if in_file is .bed or .narrowPeak>] \
  -out_file <path to output file, .npy or .csv> \
  [-layer_name <layer name to get activations from, e.g. 'flatten'>. default is output layer] \
  [--no_reverse_complement, don't evaluate on reverse complement sequences] \
  [--write_csv, write activations as .csv file instead of .npy] \
  [-score_column <output unit to extract score, e.g. 1>. use 'all' to write all units in the layer] \
  [--bayesian, do Bayesian inference with N=64 trials]

Examples:

1. Model is a binary classifier, output .csv file of probabilities for the positive class:
  [don't pass -layer_name]
  --write_csv
  (optional: --bayesian to get Bayesian predictions)

2. Model is a regression model, output .csv file of predicted values:
  [don't pass -layer_name]
  --write_csv
  (optional: --bayesian to get Bayesian predictions)

3. Model is classification or regression, output .npy file of inner-layer activations:
  -layer_name <layer_name>
  -score_column all
  [don't pass --write_csv]
```

**NOTE:** By default, reverse complement sequences are included. The output file will have twice as many activations as the input file has sequences. The order of results is:
```
pred(example_1)
pred(revcomp(example_1))
...
pred(example_n)
pred(revcomp(example_n))
```
To exclude reverse complement sequences, pass `--no_reverse_complement`.

## Authors
Heather Sestili - Implemented pipeline.  
Ziheng (Calvin) Chen - Model interpretation, SHAP, TF-MoDISCO.  
Badoi Phan - Advised on pipeline architecture. Advised on cyclic learning rate, cyclic momentum, cyclic learning rate finder, wandb integration.  
Irene Kaplow - Advised on pipeline architecture. Advised on “proportional” class weighting scheme.  
Chaitanya Srinivasan - Advised on “reciprocal” class weighting scheme.  
Spencer Gibson - Experimented with interpretation of Bayesian approximation.
