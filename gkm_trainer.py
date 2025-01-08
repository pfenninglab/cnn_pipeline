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

def _get_files_by_class(paths, targets):
    """Split files into positive and negative sets handling both dict and str paths."""
    # Extract actual file paths from potential dictionaries
    def _get_path(path_entry):
        if isinstance(path_entry, dict):
            if 'intervals' in path_entry:
                return path_entry['intervals']
            elif 'path' in path_entry:
                return path_entry['path']
            else:
                raise ValueError(f"Invalid path dictionary format: {path_entry}")
        return path_entry

    pos_files = [_get_path(p) for p, t in zip(paths, targets) if t == 1]
    neg_files = [_get_path(p) for p, t in zip(paths, targets) if t == 0]
    
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

def train_model(config):
    """Train a gkm-SVM model using config parameters."""
    # Get base output directory
    base_dir = config.dir if config.dir else os.getcwd()
    model_dir = os.path.join(base_dir, "lsgkm", config.name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if model already exists
    out_prefix = os.path.join(model_dir, config.get_run_prefix())
    model_path = f"{out_prefix}.model.txt"
    if os.path.exists(model_path):
        return model_path

    # Define combined file paths
    train_pos_fa = os.path.join(model_dir, f"{config.name}-train-pos.fa")
    train_neg_fa = os.path.join(model_dir, f"{config.name}-train-neg.fa")
    
    # Check if combined files already exist
    if os.path.exists(train_pos_fa) and os.path.exists(train_neg_fa):
        logger.info(f"Using existing combined training files:")
        logger.info(f"  Positive sequences: {train_pos_fa}")
        logger.info(f"  Negative sequences: {train_neg_fa}")
    else:
        # Get and combine training files
        pos_files, neg_files = _get_files_by_class(
            config.train_data_paths, 
            config.train_targets
        )
        
        logger.info(f"Creating combined training files:")
        logger.info("  Positive files being combined:")
        for pos_file in pos_files:
            logger.info(f"    {pos_file}")
        
        # Combine positive files
        with open(train_pos_fa, 'w') as outfile:
            for pos_file in pos_files:
                with open(pos_file) as infile:
                    outfile.write(infile.read())
        logger.info(f"  Combined positive file created: {train_pos_fa}")
                    
        logger.info("  Negative files being combined:")
        for neg_file in neg_files:
            logger.info(f"    {neg_file}")
            
        # Combine negative files  
        with open(train_neg_fa, 'w') as outfile:
            for neg_file in neg_files:
                with open(neg_file) as infile:
                    outfile.write(infile.read())
        logger.info(f"  Combined negative file created: {train_neg_fa}")

    # Calculate class weights based on sequence counts
    pos_count = count_sequences_in_fasta(train_pos_fa)
    neg_count = count_sequences_in_fasta(train_neg_fa)
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
    # Get prediction directory based on model location instead of config
    model_dir = os.path.dirname(model_path)
    pred_dir = os.path.join(model_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Generate unique prediction filename based on:
    # 1. Model basename (without .model.txt)
    # 2. Test file basename (without .fa)
    # 3. Data type and class type
    model_base = os.path.splitext(os.path.splitext(os.path.basename(model_path))[0])[0]
    test_base = os.path.splitext(os.path.basename(test_file))[0]
    pred_name = f"{model_base}.{test_base}.{data_type}-{class_type}.prediction.txt"
    pred_path = os.path.join(pred_dir, pred_name)
    
    # Count expected sequences
    seq_count = sum(1 for line in open(test_file) if line.startswith('>'))
    
    # Check if valid prediction file exists
    if os.path.exists(pred_path):
        pred_count = sum(1 for line in open(pred_path) if line.strip())
        if pred_count == seq_count:
            logger.info(f"Using existing predictions: {pred_path}")
            return pred_path
        logger.info(f"Removing incomplete predictions file: {pred_path}")
        os.remove(pred_path)  # Remove incomplete file
    
    # Run prediction
    logger.info(f"Generating new predictions: {pred_path}")
    cmd = config.get_predict_cmd(test_file, model_path, pred_path)
    if os.system(' '.join(cmd)) != 0:
        raise ValueError("Prediction failed")
        
    # Verify predictions
    if not os.path.exists(pred_path):
        raise ValueError(f"Predictions file not created: {pred_path}")
        
    pred_count = sum(1 for line in open(pred_path) if line.strip())
    if pred_count != seq_count:
        raise ValueError(f"Incomplete predictions: got {pred_count}/{seq_count}")
        
    return pred_path

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   prefix: str = '', 
                   pos_pred_file: str = None,  # Add parameters to track files
                   neg_pred_file: str = None) -> Dict[str, float]:
    """Compute binary classification metrics matching CNN pipeline.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted scores
        prefix: Optional prefix for metric names
        pos_pred_file: Path to positive predictions file used
        neg_pred_file: Path to negative predictions file used
        
    Returns:
        Dictionary of metric names and values
    """
    # Log which prediction files were used
    logger.info(f"Computing {prefix}metrics using:")
    if pos_pred_file:
        logger.info(f"  Positive predictions from: {pos_pred_file}")
    if neg_pred_file:
        logger.info(f"  Negative predictions from: {neg_pred_file}")
    
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

def evaluate_model(config: GkmConfig, model_path: str, results_path: str = None, results_dir: str = None) -> Dict[str, float]:
    """Evaluate model on validation and additional validation sets."""
    # Verify model path exists 
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
        
    # Determine results path
    if results_path:
        final_results_path = results_path
    elif results_dir:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        final_results_path = os.path.join(results_dir, f"{model_name}_results.json")
    else:
        final_results_path = f"{os.path.splitext(model_path)[0]}_results.json"
    
    # Create results directory if needed
    results_parent_dir = os.path.dirname(final_results_path)
    if results_parent_dir:
        os.makedirs(results_parent_dir, exist_ok=True)
    
    # Initialize results
    base_dir = config.dir if config.dir else os.getcwd()
    model_dir = os.path.join(base_dir, "lsgkm", config.name)
    results = {}

    # Evaluate validation set if present
    if config.val_data_paths:
        logger.info("Processing main validation set:")
        val_pos_fa = os.path.join(model_dir, f"{config.name}-val-pos.fa")
        val_neg_fa = os.path.join(model_dir, f"{config.name}-val-neg.fa")
        
        # Create combined validation files if needed
        pos_files, neg_files = _get_files_by_class(
            config.val_data_paths, config.val_targets)
            
        if not os.path.exists(val_pos_fa):
            logger.info("  Creating combined positive validation file from:")
            for f in pos_files:
                logger.info(f"    {f}")
            with open(val_pos_fa, 'w') as outfile:
                for f in pos_files:
                    with open(f) as infile:
                        outfile.write(infile.read())
        else:
            logger.info(f"  Using existing combined positive file: {val_pos_fa}")
                        
        if not os.path.exists(val_neg_fa):
            logger.info("  Creating combined negative validation file from:")
            for f in neg_files:
                logger.info(f"    {f}")
            with open(val_neg_fa, 'w') as outfile:
                for f in neg_files:
                    with open(f) as infile:
                        outfile.write(infile.read())
        else:
            logger.info(f"  Using existing combined negative file: {val_neg_fa}")
                    
        # Get validation predictions
        val_pos_pred = predict(config, model_path, val_pos_fa, 'val', 'pos')
        val_neg_pred = predict(config, model_path, val_neg_fa, 'val', 'neg')
        
        with open(val_pos_pred) as f:
            pos_pred = [float(line.split('\t')[1] if '\t' in line else line.split()[1])
                       for line in f if line.strip()]
        with open(val_neg_pred) as f:
            neg_pred = [float(line.split('\t')[1] if '\t' in line else line.split()[1])
                       for line in f if line.strip()]
            
        y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
        y_pred = np.concatenate([pos_pred, neg_pred])
        results.update(compute_metrics(y_true, y_pred, prefix='val_',
                                    pos_pred_file=val_pos_pred,
                                    neg_pred_file=val_neg_pred))

    # Evaluate additional validation sets
    if config.additional_val_data_paths:
        for idx, (paths, targets) in enumerate(zip(
            config.additional_val_data_paths,
            config.additional_val_targets
        ), 1):
            logger.info(f"Processing additional validation set {idx}:")
            add_val_pos_fa = os.path.join(model_dir, 
                f"{config.name}-val_{idx}-pos.fa")
            add_val_neg_fa = os.path.join(model_dir,
                f"{config.name}-val_{idx}-neg.fa")
                
            pos_files, neg_files = _get_files_by_class(paths, targets)
            
            if not os.path.exists(add_val_pos_fa):
                logger.info("  Creating combined positive validation file from:")
                for f in pos_files:
                    logger.info(f"    {f}")
                with open(add_val_pos_fa, 'w') as outfile:
                    for f in pos_files:
                        with open(f) as infile:
                            outfile.write(infile.read())
            else:
                logger.info(f"  Using existing combined positive file: {add_val_pos_fa}")
                            
            if not os.path.exists(add_val_neg_fa):
                logger.info("  Creating combined negative validation file from:")
                for f in neg_files:
                    logger.info(f"    {f}")
                with open(add_val_neg_fa, 'w') as outfile:
                    for f in neg_files:
                        with open(f) as infile:
                            outfile.write(infile.read())
            else:
                logger.info(f"  Using existing combined negative file: {add_val_neg_fa}")
                        
            # Get predictions
            val_pos_pred = predict(config, model_path, add_val_pos_fa, 
                                 f'val_{idx}', 'pos')
            val_neg_pred = predict(config, model_path, add_val_neg_fa,
                                 f'val_{idx}', 'neg')
            
            with open(val_pos_pred) as f:
                pos_pred = [float(line.split('\t')[1] if '\t' in line else line.split()[1])
                           for line in f if line.strip()]
            with open(val_neg_pred) as f:
                neg_pred = [float(line.split('\t')[1] if '\t' in line else line.split()[1])
                           for line in f if line.strip()]
                
            y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
            y_pred = np.concatenate([pos_pred, neg_pred])
            results.update(compute_metrics(y_true, y_pred, prefix=f'val_{idx}_',
                                        pos_pred_file=val_pos_pred,
                                        neg_pred_file=val_neg_pred))

    # Save results
    save_results(config, results, model_path, final_results_path)
    return results

def save_results(config: GkmConfig, results: Dict[str, float], model_path: str, results_path: str) -> str:
    """Save evaluation results to JSON.
    
    Args:
        config: GkmConfig object
        results: Dictionary of evaluation metrics
        model_path: Path to model file (for reference)
        results_path: Path where to save results
        
    Returns:
        Path where results were saved
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
    # Save results file
    with open(results_path, 'w') as f:
        json.dump({
            'config': vars(config),
            'model_path': model_path,
            'results': results
        }, f, indent=2)
    
    return results_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate gkm-SVM models using CNN pipeline config',
        usage='%(prog)s -config <config_file> [options]'
    )
    
    # Required config file
    parser.add_argument(
        '-config', required=True,
        help='Path to CNN pipeline YAML configuration file'
    )
    
    # Add mutually exclusive group for modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--predict',
        help='Path to input FASTA file for prediction. Enables predict-only mode'
    )
    mode_group.add_argument(
        '--model_path',
        help='Path to existing model file. Required for predict mode or to evaluate existing model'
    )
    
    # Optional outputs
    parser.add_argument(
        '--output_directory',
        help='Override base directory for relative output paths. If not provided, uses config.dir'
    )
    
    parser.add_argument(
        '--predict_output',
        help='Path for prediction output file. Only used with --predict'
    )

    parser.add_argument(
        '--results_path',
        help='Path for evaluation results JSON. If relative, uses output_directory or config.dir as base'
    )
    
    parser.add_argument(
        '--results_dir',
        help='Directory for evaluation results (ignored if results_path provided)'
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

        if args.predict:
            # Prediction mode
            if not args.model_path:
                raise ValueError("--model_path is required when using --predict")
            
            if not os.path.exists(args.predict):
                raise ValueError(f"Input file not found: {args.predict}")
            
            if not os.path.exists(args.model_path):
                raise ValueError(f"Model file not found: {args.model_path}")

            # Run prediction
            logger.info(f"Running prediction using model: {args.model_path}")
            output_file = args.predict_output or f"{os.path.splitext(args.predict)[0]}.predictions.txt"
            pred_path = predict(config, args.model_path, args.predict)
            
            # Move/rename predictions if needed
            if output_file != pred_path:
                import shutil
                shutil.move(pred_path, output_file)
                logger.info(f"Predictions saved to: {output_file}")
        else:
            # Check if we're in evaluation-only mode
            if args.model_path:
                if not os.path.exists(args.model_path):
                    raise ValueError(f"Model file not found: {args.model_path}")
                logger.info(f"Evaluation-only mode using model: {args.model_path}")
                model_path = args.model_path
            else:
                # Training mode
                logger.info("Training mode")
                model_path = train_model(config)
                logger.info(f"Model saved to {model_path}")
            
            # Evaluate model
            logger.info("Evaluating model")
            results = evaluate_model(
                config, 
                model_path,
                results_path=args.results_path,
                results_dir=args.results_dir
            )
            
            # Get final results path for logging
            if args.results_path:
                final_results_path = args.results_path
            elif args.results_dir:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                final_results_path = os.path.join(args.results_dir, f"{model_name}_results.json")
            else:
                final_results_path = f"{os.path.splitext(model_path)[0]}_results.json"
                
            logger.info(f"Results saved to {final_results_path}")
            
            # Print summary metrics
            logger.info("Summary metrics:")
            for metric, value in results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()