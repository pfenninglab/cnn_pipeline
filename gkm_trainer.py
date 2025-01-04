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

from gkm_config import GkmConfig, PathValidator, SequenceValidator

logger = logging.getLogger(__name__)

class GkmTrainerError(Exception):
    """Base exception class for gkm-trainer errors."""
    pass

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
        raise GkmTrainerError("No positive examples found")
    if not neg_files:
        raise GkmTrainerError("No negative examples found")
        
    return pos_files, neg_files

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
    
    # Use first file from each class for training
    pos_file, neg_file = pos_files[0], neg_files[0]
    
    # Get output prefix and create directory if needed
    out_prefix = config.get_run_prefix()
    output_dir = Path(out_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get training command
    cmd = config.get_train_cmd(pos_file, neg_file, out_prefix)
    
    logger.info(f"Training model with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise GkmTrainerError(f"Training failed: {e.stderr}")
    
    # Verify model file exists
    model_path = f"{out_prefix}.model.txt"
    if not os.path.exists(model_path):
        raise GkmTrainerError(f"Model file not created: {model_path}")
        
    return model_path

def predict(config: GkmConfig, model_path: str, test_file: str) -> str:
    """Run gkmpredict with configuration parameters.
    
    Args:
        config: GkmConfig object containing parameters
        model_path: Path to trained model file
        test_file: Path to test sequences FASTA file
        
    Returns:
        Path to predictions file
        
    Raises:
        GkmTrainerError: If prediction fails
    """
    # Validate input files
    PathValidator.validate_input_file(model_path)
    PathValidator.validate_input_file(test_file)
    PathValidator.validate_fasta_format(test_file)
    
    # Generate prediction output path
    pred_path = f"{os.path.splitext(model_path)[0]}_predictions.txt"
    PathValidator.validate_output_path(pred_path)
    
    # Get prediction command
    cmd = config.get_predict_cmd(test_file, model_path, pred_path)
    
    logger.info(f"Running predictions with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise GkmTrainerError(f"Prediction failed: {e.stderr}")
        
    if not os.path.exists(pred_path):
        raise GkmTrainerError(f"Predictions file not created: {pred_path}")
        
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
    """Evaluate model on training, validation and additional sets.
    
    Args:
        config: GkmConfig object with evaluation parameters
        model_path: Path to trained model file
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # Training set metrics
    pos_train, neg_train = _get_files_by_class(
        config.train_data_paths, config.train_targets)
    
    train_pred = predict(config, model_path, pos_train[0])
    with open(train_pred) as f:
        y_pred = np.array([float(line.strip()) for line in f])
    y_true = np.concatenate([
        np.ones(len(pos_train)), 
        np.zeros(len(neg_train))
    ])
    results.update(compute_metrics(y_true, y_pred))

    # Validation set metrics if available
    if config.val_data_paths:
        pos_val, neg_val = _get_files_by_class(
            config.val_data_paths, config.val_targets)
        
        val_pred = predict(config, model_path, pos_val[0]) 
        with open(val_pred) as f:
            y_pred = np.array([float(line.strip()) for line in f])
        y_true = np.concatenate([
            np.ones(len(pos_val)),
            np.zeros(len(neg_val))
        ])
        results.update(compute_metrics(y_true, y_pred, prefix='val_'))

    # Additional validation sets if available
    if config.additional_val_data_paths:
        for i, (paths, targets) in enumerate(zip(
            config.additional_val_data_paths,
            config.additional_val_targets
        ), 1):
            pos_files, neg_files = _get_files_by_class(paths, targets)
            
            pred_file = predict(config, model_path, pos_files[0])
            with open(pred_file) as f:
                y_pred = np.array([float(line.strip()) for line in f])
            y_true = np.concatenate([
                np.ones(len(pos_files)),
                np.zeros(len(neg_files))
            ])
            results.update(compute_metrics(y_true, y_pred, prefix=f'val_{i}_'))

    return results

def save_results(config: GkmConfig,
                results: Dict[str, float],
                model_path: str) -> str:
    """Save evaluation results matching CNN pipeline format."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{os.path.splitext(model_path)[0]}_results_{timestamp}.json"
    
    # Format results matching CNN pipeline
    output = {
        'model_config': {
            # Core parameters
            'word_length': config.word_length,
            'informed_cols': config.informed_cols, 
            'max_mismatch': config.max_mismatch,
            'kernel_type': config.kernel_type,
            'regularization': config.regularization,
            
            # Kernel parameters
            'gamma': config.gamma,
            'init_decay': config.init_decay,
            'half_life': config.half_life,
            
            # Runtime parameters
            'num_threads': config.num_threads,
            'use_shrinking': config.use_shrinking,
            
            # Paths
            'model_path': str(model_path)
        },
        'results': {
            k: float(f'{v:.4f}') for k, v in results.items()
        },
        'metadata': {
            'timestamp': timestamp,
            'project': config.project,
            'name': config.name
        }
    }
    
    # Save results
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    except IOError as e:
        raise GkmTrainerError(f"Failed to save results: {e}")
        
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