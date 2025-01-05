"""gkm_trainer.py: Core training and evaluation logic for gkm-SVM models.

This module handles:
1. Training gkm-SVM models using the gkmtrain executable with exact lsgkm parameters
2. Computing performance metrics matching CNN pipeline format
3. Managing validation data and predictions from CNN pipeline configs
4. Saving evaluation results in CNN-compatible format
"""

import os
import numpy as np
import subprocess
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from gkm_config import GkmConfig, Validator, Validator

logger = logging.getLogger(__name__)

class GkmTrainerError(Exception):
    """Base exception class for gkm-trainer errors."""
    pass

def get_prediction_filename(config: GkmConfig, test_file: str, prefix: str = None) -> str:
    """Generate standardized prediction filename."""
    model_base_dir = Path(config.dir) if config.dir else Path.cwd()
    pred_dir = model_base_dir / "lsgkm" / config.name / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    test_name = Path(test_file).stem
    if prefix:
        pred_path = str(pred_dir / f"{config.name}_{prefix}_{test_name}_predictions.txt")
    else:
        pred_path = str(pred_dir / f"{config.name}_{test_name}_predictions.txt")
        
    return pred_path

def count_sequences_in_fasta(fasta_file: str) -> int:
    """Count number of sequences in a FASTA file by counting header lines."""
    count = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def calculate_class_weight(pos_count: int, neg_count: int, weighting_scheme: str = 'reciprocal') -> float:
    """Calculate positive class weight based on sequence counts.
    
    Args:
        pos_count: Number of positive sequences
        neg_count: Number of negative sequences
        weighting_scheme: Strategy for weight calculation ('reciprocal' or 'proportional')
        
    Returns:
        Weight value for positive class (SVM -w parameter)
        
    Based on CNN pipeline class weighting schemes:
    - reciprocal: weight = (total_samples / num_classes) / class_count
    - proportional: weight = fraction of samples in other classes
    """
    total_count = pos_count + neg_count
    
    if weighting_scheme == 'reciprocal':
        # Chai's balancing method: weight = (N/k) / n_i
        # Where N is total samples, k is number of classes (2), n_i is class count
        balanced_count = total_count / 2
        weight = balanced_count / pos_count
        
    elif weighting_scheme == 'proportional':
        # Irene's balancing method: weight = fraction of samples in other classes
        weight = neg_count / total_count
        
    else:
        # No weighting
        weight = 1.0
        
    return weight


def train_model(config: GkmConfig) -> str:
    """Train a gkm-SVM model using config parameters.
    
    Args:
        config: GkmConfig object containing model parameters
        
    Returns:
        Path to trained model file
        
    Raises:
        GkmTrainerError: If training fails
    """
    # Resolve and validate paths
    config.resolve_paths()
    
    # Get positive and negative files
    pos_files, neg_files = config.get_train_files()
    
    # Create model directory structure
    model_base_dir = Path(config.dir) if config.dir else Path.cwd()
    model_dir = model_base_dir / "lsgkm" / config.name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    model_prefix = config.get_run_prefix()
    expected_model_path = f"{model_dir / model_prefix}.model.txt"
    if os.path.exists(expected_model_path):
        logger.info(f"Model already exists at {expected_model_path}, skipping training")
        return expected_model_path

    # Rest of the training code remains the same...
    pos_output = model_dir / f"{config.name}-pos.fa"
    neg_output = model_dir / f"{config.name}-neg.fa"
    
    logger.info(f"Combining {len(pos_files)} positive files into {pos_output}")
    with open(pos_output, 'w') as outfile:
        for pos_file in pos_files:
            with open(pos_file) as infile:
                for line in infile:
                    outfile.write(line)
                    
    logger.info(f"Combining {len(neg_files)} negative files into {neg_output}")
    with open(neg_output, 'w') as outfile:
        for neg_file in neg_files:
            with open(neg_file) as infile:
                for line in infile:
                    outfile.write(line)
    
    # Validate combined files
    Validator.validate_fasta_format(pos_output)
    Validator.validate_fasta_format(neg_output)
    Validator.validate_sequence_length(pos_output)
    Validator.validate_sequence_length(neg_output)
    
    # Calculate class weights based on sequence counts
    pos_count = count_sequences_in_fasta(pos_output)
    neg_count = count_sequences_in_fasta(neg_output)
    total_count = pos_count + neg_count
    
    logger.info(f"Training set composition:")
    logger.info(f"  Positive sequences: {pos_count:,} ({pos_count/total_count:.1%})")
    logger.info(f"  Negative sequences: {neg_count:,} ({neg_count/total_count:.1%})")
    
    # Set positive class weight if weighting scheme specified
    if hasattr(config, 'class_weight') and config.class_weight not in [None, 'none']:
        weight = calculate_class_weight(pos_count, neg_count, config.class_weight)
        logger.info(f"Using {config.class_weight} weighting scheme, positive weight = {weight:.3f}")
        config.pos_weight = weight
    else:
        logger.info("No class weighting applied")
    
    # Get output prefix and prepare command
    out_prefix = str(model_dir / config.get_run_prefix())
    cmd = config.get_train_cmd(str(pos_output), str(neg_output), out_prefix)
    cmd_str = ' '.join(cmd)
    
    logger.info(f"Training model with command: {cmd_str}")
    
    # Run training command
    exit_code = os.system(cmd_str)
    if exit_code != 0:
        raise GkmTrainerError(f"Training failed with exit code: {exit_code}")
    
    # Verify model file exists
    model_path = f"{out_prefix}.model.txt"
    if not os.path.exists(model_path):
        raise GkmTrainerError(f"Model file not created: {model_path}")
        
    return model_path


def predict(config: GkmConfig, model_path: str, test_file: str, prefix: str = None) -> np.ndarray:
    """Run prediction and return numpy array of scores."""
    pred_path = get_prediction_filename(config, test_file, prefix)
    
    # Return cached predictions if they exist
    if os.path.exists(pred_path):
        logger.info(f"Using cached predictions: {pred_path}")
        return np.loadtxt(pred_path)
        
    # Run prediction
    cmd = config.get_predict_cmd(test_file, model_path, pred_path)
    exit_code = os.system(' '.join(cmd))
    if exit_code != 0:
        raise GkmTrainerError(f"Prediction failed with exit code: {exit_code}")
        
    return np.loadtxt(pred_path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   prefix: str = '') -> Dict[str, float]:
    """Compute binary classification metrics matching CNN pipeline.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted scores
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary of metric names and values
    """
    # Verify inputs
    if len(y_true) != len(y_pred):
        raise GkmTrainerError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )
    
    if not len(y_true):
        raise GkmTrainerError("Empty prediction arrays")
        
    # Compute threshold-independent metrics
    metrics = {
        f'{prefix}auroc': roc_auc_score(y_true, y_pred),
        f'{prefix}auprc': average_precision_score(y_true, y_pred)
    }

    # Compute threshold-dependent metrics at 0.5
    y_pred_binary = (y_pred > 0.5).astype(int)
    metrics.update({
        f'{prefix}precision': precision_score(y_true, y_pred_binary),
        f'{prefix}recall': recall_score(y_true, y_pred_binary),
        f'{prefix}f1': f1_score(y_true, y_pred_binary)
    })

    return metrics


def evaluate_model(config: GkmConfig, model_path: str) -> Dict[str, float]:
    """Evaluate model on training, validation and additional sets."""
    results = {}
    
    # Training set metrics
    pos_train, neg_train = config.get_train_files()
        
    # Count sequences in each file
    pos_count = sum(count_sequences_in_fasta(f) for f in pos_train)
    neg_count = sum(count_sequences_in_fasta(f) for f in neg_train)
    
    # Predict on all files
    all_predictions = []
    for file in pos_train:
        pred_file = predict(config, model_path, file, "train-pos")
        predictions = parse_prediction_file(pred_file)
        all_predictions.extend(predictions)
    for file in neg_train:
        pred_file = predict(config, model_path, file, "train-neg")
        predictions = parse_prediction_file(pred_file)
        all_predictions.extend(predictions)
    
    y_pred = np.array(all_predictions)
    y_true = np.concatenate([
        np.ones(pos_count),
        np.zeros(neg_count)
    ])
    
    results.update(compute_metrics(y_true, y_pred))

    # Validation set metrics if available
    if config.val_data_paths:
        pos_val, neg_val = _get_files_by_class(
            config.val_data_paths, config.val_targets)
            
        # Count sequences in validation files
        pos_count = sum(count_sequences_in_fasta(f) for f in pos_val)
        neg_count = sum(count_sequences_in_fasta(f) for f in neg_val)
        
        # Predict on all validation files
        all_predictions = []
        for file in pos_val:
            pred_file = predict(config, model_path, file, "validation-pos")
            predictions = parse_prediction_file(pred_file)
            all_predictions.extend(predictions)
        for file in neg_val:
            pred_file = predict(config, model_path, file, "validation-neg")
            predictions = parse_prediction_file(pred_file)
            all_predictions.extend(predictions)
        
        y_pred = np.array(all_predictions)
        y_true = np.concatenate([
            np.ones(pos_count),
            np.zeros(neg_count)
        ])
            
        results.update(compute_metrics(y_true, y_pred, prefix='val_'))

    # Additional validation sets if available
    if config.additional_val_data_paths:
        for i, (paths, targets) in enumerate(zip(
            config.additional_val_data_paths,
            config.additional_val_targets
        ), 1):
            pos_files, neg_files = _get_files_by_class(paths, targets)
            
            # Count sequences in additional validation files
            pos_count = sum(count_sequences_in_fasta(f) for f in pos_files)
            neg_count = sum(count_sequences_in_fasta(f) for f in neg_files)
            
            # Predict on all files in this validation set
            all_predictions = []
            for file in pos_files:
                pred_file = predict(config, model_path, file, f"additional_validation_{i}-pos")
                predictions = parse_prediction_file(pred_file)
                all_predictions.extend(predictions)
            for file in neg_files:
                pred_file = predict(config, model_path, file, f"additional_validation_{i}-neg")
                predictions = parse_prediction_file(pred_file)
                all_predictions.extend(predictions)
            
            y_pred = np.array(all_predictions)
            y_true = np.concatenate([
                np.ones(pos_count),
                np.zeros(neg_count)
            ])
                
            results.update(compute_metrics(y_true, y_pred, prefix=f'val_{i}_'))

    return results

def save_results(config: GkmConfig,
                results: Dict[str, float],
                model_path: str) -> str:
    """Save evaluation results using standardized naming."""
    output_path = f"{os.path.splitext(model_path)[0]}_results.json"
    
    output = {
        'model_config': config.__dict__,
        'results': {k: float(f'{v:.4f}') for k, v in results.items()},
        'metadata': {
            'project': config.project,
            'name': config.name
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
        
    return output_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and predict with gkm-SVM models using CNN pipeline config',
        usage='%(prog)s -config <config_file> [options]'
    )
    
    # Required config file
    parser.add_argument(
        '-config', required=True,
        help='Path to CNN pipeline YAML configuration file'
    )
    
    # Prediction mode arguments
    parser.add_argument(
        '--predict',
        help='Run in prediction mode with given input FASTA/BED file'
    )
    parser.add_argument(
        '--predict_output',
        help='Output file for predictions (default: <model_prefix>_predictions.txt)'
    )
    
        # Core gkm-SVM parameters (matching gkmtrain)
    parser.add_argument(
        '-t', type=int, choices=[0, 1, 2, 3, 4, 5],
        help='set kernel function (default: 4 wgkm)'
    )
    parser.add_argument(
        '-l', type=int,
        help='set word length, 3<=l<=12 (default: 11)'
    )
    parser.add_argument(
        '-k', type=int,
        help='set number of informative column, k<=l (default: 7)'
    )
    parser.add_argument(
        '-d', type=int,
        help='set maximum number of mismatches to consider, d<=4 (default: 3)'
    )
    
    # Kernel-specific parameters
    parser.add_argument(
        '-g', type=float,
        help='set gamma for RBF kernel. -t 3 or 5 only (default: 1.0)'
    )
    parser.add_argument(
        '-M', type=int,
        help='set the initial value (M) of the exponential decay function\n'
             'for wgkm-kernels. max=255, -t 4 or 5 only (default: 50)'
    )
    parser.add_argument(
        '-H', type=float,
        help='set the half-life parameter (H) that is the distance (D) required\n'
             'to fall to half of its initial value in the exponential decay\n'
             'function for wgkm-kernels. -t 4 or 5 only (default: 50)'
    )
    
    # SVM parameters
    parser.add_argument(
        '-c', type=float,
        help='set the regularization parameter SVM-C (default: 1.0)'
    )
    parser.add_argument(
        '-e', type=float,
        help='set the precision parameter epsilon (default: 0.001)'
    )
    parser.add_argument(
        '-w', type=float,
        help='set the parameter SVM-C to w*C for the positive set (default: 1.0)'
    )
    parser.add_argument(
        '-m', type=float,
        help='set cache memory size in MB (default: 100.0)'
    )
    parser.add_argument(
        '-s', action='store_true',
        help='if set, use the shrinking heuristics'
    )
    
    # Runtime parameters
    parser.add_argument(
        '-v', type=int, default=2, choices=[0, 1, 2, 3, 4],
        help='set the level of verbosity (default: 2)\n'
             '  0 -- error msgs only (ERROR)\n'
             '  1 -- warning msgs (WARN)\n'
             '  2 -- progress msgs at coarse-grained level (INFO)\n'
             '  3 -- progress msgs at fine-grained level (DEBUG)\n'
             '  4 -- progress msgs at finer-grained level (TRACE)'
    )
    parser.add_argument(
        '-T', type=int, default=1, choices=[1, 4, 16],
        help='set the number of threads for parallel calculation, 1, 4, or 16\n'
             '(default: 1)'
    )
    
    # CNN pipeline specific overrides
    parser.add_argument(
        '--model_dir',
        help='Override model directory from config'
    )
    parser.add_argument(
        '--output_prefix',
        help='Override output prefix from config'
    )
    parser.add_argument(
        '--genome_path',
        help='Path to genome file for BED conversion'
    )
    
    return parser.parse_args()

def update_config_from_args(config: GkmConfig, args: argparse.Namespace) -> GkmConfig:
    """Update configuration with command line arguments.
    
    Args:
        config: Original GkmConfig object from YAML
        args: Parsed command line arguments
        
    Returns:
        Updated GkmConfig object
    """
    # Map of argument names to config attributes
    param_map = {
        't': 'kernel_type',
        'l': 'word_length',
        'k': 'informed_cols',
        'd': 'max_mismatch',
        'g': 'gamma',
        'M': 'init_decay',
        'H': 'half_life',
        'c': 'regularization',
        'e': 'epsilon',
        'w': 'pos_weight',
        'm': 'cache_memory',
        's': 'use_shrinking',
        'v': 'verbosity',
        'T': 'num_threads'
    }
    
    # Update config with non-None argument values
    for arg_name, config_name in param_map.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, config_name, value)
    
    # Handle special CNN pipeline parameters
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.output_prefix:
        config.output_prefix = args.output_prefix
    if args.genome_path:
        config.genome_path = args.genome_path
    
    return config

def setup_logging(verbosity: int):
    """Configure logging based on verbosity level."""
    log_levels = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
        4: logging.DEBUG  # TRACE maps to DEBUG as Python lacks TRACE
    }
    
    logging.basicConfig(
        level=log_levels[verbosity],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.v)
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate config
        logger.info(f"Loading configuration from {args.config}")
        config = GkmConfig.from_yaml(args.config)
        config = update_config_from_args(config, args)
        config.validate()
        
        if args.predict:
            # Prediction mode
            if not os.path.exists(args.predict):
                raise GkmTrainerError(f"Input file not found: {args.predict}")
                
            # Find latest model file if not specified
            model_dir = config.model_dir or os.getcwd()
            model_files = sorted(
                Path(model_dir).glob("*.model.txt"),
                key=os.path.getmtime
            )
            if not model_files:
                raise GkmTrainerError(f"No model files found in {model_dir}")
            model_path = str(model_files[-1])
            
            # Run prediction
            logger.info(f"Running prediction using model: {model_path}")
            output_file = args.predict_output or f"{os.path.splitext(model_path)[0]}_predictions.txt"
            pred_path = predict(config, model_path, args.predict)
            
            # Move/rename predictions if needed
            if output_file != pred_path:
                import shutil
                shutil.move(pred_path, output_file)
                logger.info(f"Predictions saved to: {output_file}")
        else:
            # Training mode
            logger.info("Training model")
            model_path = train_model(config)
            logger.info(f"Model saved to {model_path}")
            
            logger.info("Evaluating model")
            results = evaluate_model(config, model_path)
            results_path = save_results(config, results, model_path)
            logger.info(f"Results saved to {results_path}")
            
            # Print summary metrics
            logger.info("Summary metrics:")
            for metric, value in results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()