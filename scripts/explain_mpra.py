print("loading packages")
# Allow importing from one directory up
import sys
sys.path.append('..')
import shap
import numpy as np
import os

import models
print("loaded packages")

def explain(args):
    print("___________________getting data___________________")
    bg, fg = get_data(args.bg_data_paths, args.fg_data_paths)
    print("___________________got data, getting shap___________________")
    shap_values = get_deepshap_scores(bg, fg, args.interp_model_path)
    print("___________________got shap, saving shap values___________________")
    save_data(fg, shap_values, args.report_dir)
    print("___________________saved shap values, running modisco___________________")

def get_data(bg_data_paths, fg_data_paths):
    import numpy as np
    from dataset import SequenceTfDataset

    bg_data = SequenceTfDataset(source_files=[bg_data_paths],targets=[0],targets_are_classes=False,endless=False)

    fg_data = SequenceTfDataset(source_files=[fg_data_paths],targets=[0],targets_are_classes=False,endless=False)

    # Get all validation data as arrays
    fg_sequences, fg_targets = fg_data.dataset
    print("Shape of one-hot encoded sequences:", fg_sequences.shape)
    bg_sequences, bg_targets = bg_data.dataset
    print("Shape of one-hot encoded sequences:", bg_sequences.shape)

    return bg_sequences, fg_sequences 

def get_deepshap_scores(bg, fg, interp_model_path):
    model = models.load_model(interp_model_path)
    explainer = shap.DeepExplainer(model, bg)
    shap_values = explainer.shap_values(fg)

    # For regression models, shap_values is an array, not a list
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return shap_values

def save_data(fg, shap_values, report_dir):
    # Convert data to length-last format (num_examples, 4, sequence_length)
    # Transpose from (num_examples, sequence_length, 4) to (num_examples, 4, sequence_length)
    fg_length_last = np.transpose(fg, (0, 2, 1))
    shap_values = shap_values.squeeze(-1)
    shap_values_length_last = np.transpose(shap_values, (0, 2, 1))

    # Save the sequences and attributions to .npz files
    os.makedirs(report_dir, exist_ok=True)
    np.savez_compressed(os.path.join(report_dir, 'onehot_encoded_sequences.npz'), fg_length_last)
    np.savez_compressed(os.path.join(report_dir, 'attributions_from_shap.npz'), shap_values_length_last)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bg_data_paths', type=str, required=True, help='Path to background data file')
    parser.add_argument('-fg_data_paths', type=str, required=True, help='Path to foreground data file')
    parser.add_argument('-report_dir', type=str, required=True, help='Directory to save the report')
    parser.add_argument('-interp_model_path', type=str, required=True, help='Path to the interpretation model')
    return parser.parse_args()

if __name__ == '__main__':
    explain(get_args())
