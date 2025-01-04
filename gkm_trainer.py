"""gkm_trainer.py: Core training and evaluation logic for gkm-SVM models.

This module handles:
1. Training gkm-SVM models using the gkmtrain executable with exact lsgkm parameters
2. Computing performance metrics matching CNN pipeline format
3. Managing validation data and predictions
4. Saving evaluation results in CNN-compatible format
"""

import os
import numpy as np
import subprocess
import logging
import json
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