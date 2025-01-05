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

from gkm_config import GkmConfig, validate_fasta_file
from gkm_config import calculate_class_weight

logger = logging.getLogger(__name__)

def _get_files_by_class(paths: List[str], targets: List[int]) -> Tuple[List[str], List[str]]:
    """Split files into positive and negative sets.
    
    Args:
        paths: List of file paths
        targets: List of binary targets (0/1)
        
    Returns:
        Tuple of (positive_files, negative_files)
    """
    pos_files = [p for p, t in zip(paths, targets) if t == 1]
    neg_files = [p for p, t in zip(paths, targets) if t == 0]
    
    if not pos_files:
        raise ValueError("No positive examples found")
    if not neg_files:
        raise ValueError("No negative examples found")
        
    return pos_files, neg_files

def count_sequences_in_fasta(fasta_file: str) -> int:
    """Count number of sequences in a FASTA file by counting header lines."""
    count = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def train_model(config: GkmConfig) -> str:
    """Train a gkm-SVM model using config parameters."""
    # Get base output directory
    base_dir = config.dir if config.dir else os.getcwd()
    model_dir = os.path.join(base_dir, "lsgkm", config.name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define combined file paths
    train_pos_fa = os.path.join(model_dir, f"{config.name}-train-pos.fa")
    train_neg_fa = os.path.join(model_dir, f"{config.name}-train-neg.fa")
    
    # Check if combined files already exist
    if not (os.path.exists(train_pos_fa) and os.path.exists(train_neg_fa)):
        # Get and combine training files
        pos_files, neg_files = _get_files_by_class(
            config.train_data_paths, 
            config.train_targets
        )
        
        # Combine positive files
        with open(train_pos_fa, 'w') as outfile:
            for pos_file in pos_files:
                with open(pos_file) as infile:
                    outfile.write(infile.read())
                    
        # Combine negative files  
        with open(train_neg_fa, 'w') as outfile:
            for neg_file in neg_files:
                with open(neg_file) as infile:
                    outfile.write(infile.read())
    
    # Get output prefix and prepare command
    out_prefix = os.path.join(model_dir, config.get_run_prefix())
    cmd = config.get_train_cmd(train_pos_fa, train_neg_fa, out_prefix)
    cmd_str = ' '.join(cmd)
    
    # Run training command
    exit_code = os.system(cmd_str)
    if exit_code != 0:
        raise ValueError(f"Training failed with exit code: {exit_code}")
        
    # Verify model file exists
    model_path = f"{out_prefix}.model.txt"
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not created: {model_path}")
        
    return model_path


def predict(config: GkmConfig, model_path: str, test_file: str,
           data_type: str = 'train', class_type: str = 'pos') -> str:
    """Run gkmpredict with configuration parameters."""
    # Get prediction directory
    base_dir = config.dir if config.dir else os.getcwd()
    model_dir = os.path.join(base_dir, "lsgkm", config.name)
    pred_dir = os.path.join(model_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Generate prediction filename based on model name and data type
    model_base = os.path.splitext(os.path.basename(model_path))[0]
    pred_name = f"{model_base}-{data_type}-{class_type}.prediction.txt"
    pred_path = os.path.join(pred_dir, pred_name)
    
    # Check if valid prediction file already exists
    if os.path.exists(pred_path):
        # Verify file has content and correct format
        try:
            with open(pred_path) as f:
                first_line = f.readline().strip()
                # Check if file has tab-delimited format with score in second column
                if first_line and len(first_line.split('\t')) >= 2:
                    try:
                        float(first_line.split('\t')[1])  # Verify score is float
                        return pred_path  # File exists and is valid
                    except ValueError:
                        pass  # Score not valid float, will recreate file
        except:
            pass  # File not readable or empty, will recreate
    
    # Run prediction command
    cmd = config.get_predict_cmd(test_file, model_path, pred_path)
    cmd_str = ' '.join(cmd)
    
    exit_code = os.system(cmd_str)
    if exit_code != 0:
        raise ValueError(f"Prediction failed with exit code: {exit_code}")
        
    # Verify prediction file exists and has correct format
    if not os.path.exists(pred_path):
        raise ValueError(f"Predictions file not created: {pred_path}")
        
    return pred_path


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
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )
    
    if not len(y_true):
        raise ValueError("Empty prediction arrays")
        
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
    base_dir = config.dir if config.dir else os.getcwd()
    model_dir = os.path.join(base_dir, "lsgkm", config.name)
    results = {}

    # Evaluate training set
    train_pos_fa = os.path.join(model_dir, f"{config.name}-train-pos.fa")
    train_neg_fa = os.path.join(model_dir, f"{config.name}-train-neg.fa")
    
    # Get training predictions
    train_pos_pred = predict(config, model_path, train_pos_fa, 'train', 'pos')
    train_neg_pred = predict(config, model_path, train_neg_fa, 'train', 'neg')
    
    # Load predictions
    with open(train_pos_pred) as f:
        pos_pred = [float(line.split('\t')[1]) for line in f]
    with open(train_neg_pred) as f:
        neg_pred = [float(line.split('\t')[1]) for line in f]
        
    # Calculate training metrics
    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    y_pred = np.concatenate([pos_pred, neg_pred])
    results.update(compute_metrics(y_true, y_pred, prefix=''))

    # Evaluate validation set
    if config.val_data_paths:
        # Create combined validation files
        val_pos_fa = os.path.join(model_dir, f"{config.name}-val-pos.fa")
        val_neg_fa = os.path.join(model_dir, f"{config.name}-val-neg.fa")
        
        pos_files, neg_files = _get_files_by_class(
            config.val_data_paths, config.val_targets)
            
        # Combine validation files
        with open(val_pos_fa, 'w') as outfile:
            for f in pos_files:
                with open(f) as infile:
                    outfile.write(infile.read())
        with open(val_neg_fa, 'w') as outfile:
            for f in neg_files:
                with open(f) as infile:
                    outfile.write(infile.read())
                    
        # Get validation predictions
        val_pos_pred = predict(config, model_path, val_pos_fa, 'val', 'pos')
        val_neg_pred = predict(config, model_path, val_neg_fa, 'val', 'neg')
        
        with open(val_pos_pred) as f:
            pos_pred = [float(line.split('\t')[1]) for line in f]
        with open(val_neg_pred) as f:
            neg_pred = [float(line.split('\t')[1]) for line in f]
            
        y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
        y_pred = np.concatenate([pos_pred, neg_pred])
        results.update(compute_metrics(y_true, y_pred, prefix='val_'))

    # Evaluate additional validation sets
    if config.additional_val_data_paths:
        for idx, (paths, targets) in enumerate(zip(
            config.additional_val_data_paths,
            config.additional_val_targets
        ), 1):
            # Create combined files for this validation set
            add_val_pos_fa = os.path.join(model_dir, 
                f"{config.name}-val_{idx}-pos.fa")
            add_val_neg_fa = os.path.join(model_dir,
                f"{config.name}-val_{idx}-neg.fa")
                
            pos_files, neg_files = _get_files_by_class(paths, targets)
            
            # Combine files
            with open(add_val_pos_fa, 'w') as outfile:
                for f in pos_files:
                    with open(f) as infile:
                        outfile.write(infile.read())
            with open(add_val_neg_fa, 'w') as outfile:
                for f in neg_files:
                    with open(f) as infile:
                        outfile.write(infile.read())
                        
            # Get predictions
            val_pos_pred = predict(config, model_path, add_val_pos_fa, 
                                 f'val_{idx}', 'pos')
            val_neg_pred = predict(config, model_path, add_val_neg_fa,
                                 f'val_{idx}', 'neg')
            
            with open(val_pos_pred) as f:
                pos_pred = [float(line.split('\t')[1]) for line in f]
            with open(val_neg_pred) as f:
                neg_pred = [float(line.split('\t')[1]) for line in f]
                
            y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
            y_pred = np.concatenate([pos_pred, neg_pred])
            results.update(compute_metrics(y_true, y_pred, prefix=f'val_{idx}_'))

    return results

def save_results(config: GkmConfig, results: Dict[str, float], model_path: str) -> str:
    """Save evaluation results to JSON."""
    output_path = f"{os.path.splitext(model_path)[0]}_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'config': vars(config),
            'results': results
        }, f, indent=2)
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
                raise ValueError(f"Input file not found: {args.predict}")
                
            # Find latest model file if not specified
            model_dir = config.model_dir or os.getcwd()
            model_files = sorted(
                Path(model_dir).glob("*.model.txt"),
                key=os.path.getmtime
            )
            if not model_files:
                raise ValueError(f"No model files found in {model_dir}")
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