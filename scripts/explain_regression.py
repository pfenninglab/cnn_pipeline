# explain.py: Get SHAP importance scores and run MoDISco-lite.

import sys
sys.path.append('..')

import argparse
import os
import re
import subprocess

import numpy as np
print("___________________importing SHAP___________________")
import shap
print("___________________importing wandb___________________")
import wandb

print("___________________importing utils___________________")
import models
import utils
from dataset import SequenceTfDataset

def explain(args):
    init(args)
    print("___________________Getting data___________________")
    bg, fg = get_data()
    print("___________________Calculating DeepSHAP scores___________________")
    shap_values = get_deepshap_scores(bg, fg)
    print("___________________Saving SHAP values___________________")
    save_data(fg, shap_values)
    print("___________________Running MoDISco___________________")
    run_modisco()
    print("___________________Generating reports___________________")
    generate_reports()
    #correct_report_html()
    print("___________________Job completed___________________")

def init(args):
    config, project = utils.get_config(args.config)
    wandb.init(config=config, project=project, mode="disabled")
    utils.validate_config(wandb.config)

def get_data():
    # Background is the training set
    bg_data = SequenceTfDataset(
        wandb.config.shap_bg_data_paths,
        wandb.config.shap_bg_targets,
        targets_are_classes=False,
    )

    # Foreground is sequences with signal values larger than the threshold from the validation set
    val_data = SequenceTfDataset(
        wandb.config.shap_fg_data_paths,
        wandb.config.shap_fg_targets,
        targets_are_classes=False,
    )

    # Get all validation data as arrays
    val_sequences, val_targets = val_data.get_subset_as_arrays(len(val_data))

    # Convert targets to numpy array of floats
    val_targets = np.array(val_targets).astype(float)

    print("Foreground signal threshold:", wandb.config.shap_pos_value)
    if wandb.config.shap_pos_value != 0:
        if wandb.config.shap_pos_value > 0:
            fg_idx = val_targets > wandb.config.shap_pos_value
        else:
            fg_idx = val_targets < wandb.config.shap_pos_value
        fg_sequences = val_sequences[fg_idx]
    else:
        fg_sequences = val_sequences

    # Check if we have enough foreground sequences, if not, use all.
    if len(fg_sequences) < wandb.config.shap_num_fg:
        print(f"Not enough foreground sequences with target threshold of {wandb.config.shap_pos_value}. Found {len(fg_sequences)} but need {wandb.config.shap_num_fg}.")
        fg = fg_sequences
        print(f"Use {len(fg)} foreground sequences that pass target threshold.")
    else:
        # Randomly select shap_num_fg sequences from fg_sequences
        fg_indices = np.random.choice(len(fg_sequences), size=wandb.config.shap_num_fg, replace=False)
        fg = fg_sequences[fg_indices]

    # For background data, get a subset if need fewer sequences
    if wandb.config.shap_num_bg < len(bg_data):
        bg, _ = bg_data.get_subset_as_arrays(wandb.config.shap_num_bg)
    else:
        bg, _ = bg_data.get_subset_as_arrays(len(bg_data))

    return bg, fg

def get_deepshap_scores(bg, fg):
    model = models.load_model(wandb.config.interp_model_path)
    explainer = shap.DeepExplainer(model, bg)
    shap_values = explainer.shap_values(fg)

    # For regression models, shap_values is an array, not a list
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return shap_values

def save_data(fg, shap_values):
    # Convert data to length-last format (num_examples, 4, sequence_length)
    report_dir = wandb.config.get('modisco_report_dir', 'report')
    fg_length_last = np.transpose(fg, (0, 2, 1))
    shap_values = shap_values.squeeze(-1)
    shap_values_length_last = np.transpose(shap_values, (0, 2, 1))

    # Save the sequences and attributions to .npz files
    np.savez_compressed(os.path.join(report_dir, 'onehot_encoded_sequences.npz'), fg_length_last)
    np.savez_compressed(os.path.join(report_dir, 'attributions_from_shap.npz'), shap_values_length_last)

def run_modisco():
    report_dir = wandb.config.get('modisco_report_dir', 'report')
    max_seqlets_per_metacluster = wandb.config.get('modisco_max_seqlets', 100000)
    output_h5 = wandb.config.get('modisco_output', 'modisco_results.h5')
    cmd = [
        'modisco',
        'motifs',
        '-s', os.path.join(report_dir, 'onehot_encoded_sequences.npz'),
        '-a', os.path.join(report_dir, 'attributions_from_shap.npz'),
        '-n', str(max_seqlets_per_metacluster),
        '-o', output_h5
    ]
    print('Running MoDISco-lite command:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def generate_reports():
    try:
        output_h5 = wandb.config.get('modisco_output', 'modisco_results.h5')
        report_dir = wandb.config.get('modisco_report_dir', 'report')
        motifs_database = wandb.config.get('modisco_motifs_database', None)

        # Ensure report_dir ends with '/'
        report_dir = report_dir.rstrip('/') + '/'

        # Create the report directory if it doesn't exist
        os.makedirs(report_dir, exist_ok=True)
        
        cmd = [
            'modisco',
            'report',
            '-i', output_h5,
            '-o', './',
            '-s', './'
        ]

        # If a motifs database is specified, include it
        if motifs_database:
            cmd.extend(['-m', motifs_database])

        print('Changing directory to:', report_dir)
        os.chdir(report_dir)

        print('Generating report with command:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def correct_report_html():
    report_dir = wandb.config.get('modisco_report_dir', 'report')
    report_path = os.path.join(report_dir, 'motifs.html')

    # Extract the absolute directory path of the report
    report_dir_abs = os.path.dirname(os.path.abspath(report_path))

    # Load the HTML file content
    with open(report_path, "r") as file:
        html_content = file.read()

    # Use regex to replace occurrences of the absolute path
    html_content = re.sub(re.escape(report_dir_abs) + r'(/[^"]*)', r'.\1', html_content)

    # Save the modified HTML file
    with open(os.path.join(report_dir, "motifs_new.html"), "w") as file:
        file.write(html_content)

    print("Relative path replacement complete. Modified report saved as 'motifs_new.html'.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    explain(get_args())
