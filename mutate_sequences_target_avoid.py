#!/usr/bin/env python3
"""
Modified mutation script for target vs avoid model optimization.
Uses reward function: e^(target_model_prediction - avoid_model_prediction)
Removes cross-species validation and focuses on two-model optimization.

Usage:
    python mutate_sequences_target_avoid.py --target <target_cell_type> --avoid <avoid_cell_type> [options]

Example:
    python mutate_sequences_target_avoid.py --target GLUT6 --avoid GLUT5 --iterations 20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_ind, mannwhitneyu
import time
import os
import dill
import importlib
import sys
import argparse
import tensorflow as tf
import wandb
from tqdm import tqdm

# Add current directory to path for local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure GPU memory growth for better memory management
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def get_gpu_memory_info():
    """Get GPU memory information for batch size optimization."""
    try:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        if 'device_memory_size' in gpu_details:
            memory_gb = gpu_details['device_memory_size'] / (1024**3)
            print(f"GPU memory: {memory_gb:.1f} GB")
            return memory_gb
    except:
        pass
    return None

# Import local helpers instead of external synthESizer
import helpers as seq_functions
import helpers as saturation_mutagenesis_functions
from helpers import adalead_onehot as adalead
from models import predict_with_uncertainty

# Configuration
bigboy_or_lane = 'local'  # Use local execution

# Path settings
if bigboy_or_lane == 'bigboy':
    dh_prefix = '/lane/dorsalhorn/'
    home_prefix = '/lane/home/'
elif bigboy_or_lane == 'local':
    # Use current directory structure
    dh_prefix = './data/'
    home_prefix = './'
else:
    dh_prefix = '/projects/pfenninggroup/singleCell/Macaque_SealDorsalHorn_snATAC-seq/'
    home_prefix = '/home/mleone2/'

datapath = dh_prefix + 'data/tidy_data/synthetic_design/testing'
figurepath = dh_prefix + 'figures/exploratory/synthetic_design/testing'

# Create directories if they don't exist
os.makedirs(datapath, exist_ok=True)
os.makedirs(figurepath, exist_ok=True)

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================

def init_wandb(target_cell_type, avoid_cell_type, args):
    """
    Initialize wandb for tracking mutation optimization.
    
    Args:
        target_cell_type: Name of target cell type
        avoid_cell_type: Name of avoid cell type
        args: Command line arguments
        
    Returns:
        wandb run object or None if initialization fails
    """
    try:
        # Use the same project as cnn_pipeline training scripts
        project = "cnn_pipeline_cortex_enhancers"
        entity = "isuru-herath10-carnegie-mellon-university"  # Your personal account
        
        # Create run name
        run_name = f"mutation_script_{target_cell_type}_vs_{avoid_cell_type}"
        
        # Create config dictionary
        config = {
            'target_cell_type': target_cell_type,
            'avoid_cell_type': avoid_cell_type,
            'iterations': args.iterations,
            'seq_length': args.seq_length,
            'n_seqs': args.n_seqs,
            'adalead_rounds': args.adalead_rounds,
            'adalead_recombine_turns': args.adalead_recombine_turns,
            'batch_size': args.batch_size,
            'output_prefix': args.output_prefix,
            'reward_function': 'e^(target_prediction - avoid_prediction)',
            'optimization_method': 'AdaLead + Saturation Mutagenesis'
        }
        
        # Initialize wandb
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            tags=['mutation_optimization', 'target_avoid', f'{target_cell_type}_vs_{avoid_cell_type}']
        )
        
        print(f"✅ Wandb initialized successfully: {run_name}")
        return wandb
        
    except Exception as e:
        print(f"⚠️  Wandb initialization failed: {e}")
        print("   Continuing without wandb tracking...")
        return None

def log_reward_metrics(step, reward_type, rewards, target_preds=None, avoid_preds=None):
    """
    Log reward metrics to wandb.
    
    Args:
        step: Current step/iteration
        reward_type: Type of reward ('initial', 'adalead', 'saturation_mutagenesis')
        rewards: Array of reward values
        target_preds: Target model predictions (optional)
        avoid_preds: Avoid model predictions (optional)
    """
    if wandb.run is None:
        return  # Skip logging if wandb is not initialized
    
    metrics = {
        f'{reward_type}/reward_mean': np.mean(rewards),
        f'{reward_type}/reward_std': np.std(rewards),
        f'{reward_type}/reward_max': np.max(rewards),
        f'{reward_type}/reward_min': np.min(rewards),
        f'{reward_type}/step': step
    }
    
    if target_preds is not None:
        metrics[f'{reward_type}/target_pred_mean'] = np.mean(target_preds)
        metrics[f'{reward_type}/target_pred_std'] = np.std(target_preds)
        metrics[f'{reward_type}/target_pred_max'] = np.max(target_preds)
        metrics[f'{reward_type}/target_pred_min'] = np.min(target_preds)
    
    if avoid_preds is not None:
        metrics[f'{reward_type}/avoid_pred_mean'] = np.mean(avoid_preds)
        metrics[f'{reward_type}/avoid_pred_std'] = np.std(avoid_preds)
        metrics[f'{reward_type}/avoid_pred_max'] = np.max(avoid_preds)
        metrics[f'{reward_type}/avoid_pred_min'] = np.min(avoid_preds)
    
    wandb.log(metrics)

# =============================================================================
# MODEL PATH DICTIONARY
# =============================================================================

# Dictionary mapping cell type combinations to model paths
# TODO: Fill in actual model paths
MODEL_PATHS = {
    'L3.IT_vs_L4.IT': {
        'target': '/home/arb1/cnn_pipeline/wandb/run-20251027_175203-oza0mj0j-Corces_Enh_L3.IT_fold1_first/files/model-best.h5',
        'avoid': '/home/arb1/cnn_pipeline/wandb/run-20251027_180454-0njg66rn-Corces_Enh_L4.IT_fold1_first/files/model-best.h5'
    },
    # Add more cell type combinations as needed
}

def get_model_paths(target_cell_type, avoid_cell_type):
    """
    Get model paths for target and avoid cell types.
    
    Args:
        target_cell_type: Name of target cell type
        avoid_cell_type: Name of avoid cell type
        
    Returns:
        Tuple of (target_model_path, avoid_model_path)
    """
    key = f"{target_cell_type}_vs_{avoid_cell_type}"
    
    if key not in MODEL_PATHS:
        raise ValueError(f"Model paths not found for {target_cell_type} vs {avoid_cell_type}. "
                        f"Available combinations: {list(MODEL_PATHS.keys())}")
    
    paths = MODEL_PATHS[key]
    return paths['target'], paths['avoid']

# =============================================================================
# TARGET VS AVOID REWARD FUNCTION
# =============================================================================

def target_avoid_reward(target_predictions, avoid_predictions):
    """
    Calculate reward using target vs avoid model predictions.
    Reward = e^(target_prediction - avoid_prediction)
    
    Args:
        target_predictions: Predictions from target model
        avoid_predictions: Predictions from avoid model
        
    Returns:
        Reward values (higher is better)
    """
    return np.exp(target_predictions - avoid_predictions)

def load_target_avoid_models(target_model_path, avoid_model_path):
    """
    Load target and avoid models with input validation.
    
    Args:
        target_model_path: Path to target model (.h5 file)
        avoid_model_path: Path to avoid model (.h5 file)
        
    Returns:
        Dictionary with 'target' and 'avoid' models
    """
    models = {}
    
    # Load target model
    print(f"Loading target model from: {target_model_path}")
    if not os.path.exists(target_model_path):
        raise FileNotFoundError(f"Target model not found: {target_model_path}")
    
    # Load avoid model
    print(f"Loading avoid model from: {avoid_model_path}")
    if not os.path.exists(avoid_model_path):
        raise FileNotFoundError(f"Avoid model not found: {avoid_model_path}")
    
    # Define custom objects that might be in the saved models
    # Import custom metrics from the cnn_pipeline codebase
    try:
        from metrics import PearsonCorrelation, SpearmanCorrelation, MulticlassMetric
        from lr_schedules import ClrScaleFn
        custom_objects = {
            'PearsonCorrelation': PearsonCorrelation,
            'SpearmanCorrelation': SpearmanCorrelation,
            'MulticlassMetric': MulticlassMetric,
            'pearson_correlation': PearsonCorrelation,
            'spearman_correlation': SpearmanCorrelation,
            'scale_fn': ClrScaleFn.scale_fn
        }
    except ImportError:
        print("Warning: Could not import custom metrics, using None for custom objects")
        custom_objects = {
            'PearsonCorrelation': None,
            'pearson_correlation': None,
            'PearsonCorrelationMetric': None,
            'pearson_correlation_metric': None,
            'scale_fn': None
        }
    
    # Try to load models with custom objects
    try:
        models['target'] = tf.keras.models.load_model(target_model_path, custom_objects=custom_objects)
        models['avoid'] = tf.keras.models.load_model(avoid_model_path, custom_objects=custom_objects)
        print("Models loaded successfully with custom objects")
    except Exception as e:
        print(f"Failed to load with custom objects: {e}")
        print("Trying to load without custom objects...")
        
        # If that fails, try loading without custom objects
        try:
            models['target'] = tf.keras.models.load_model(target_model_path, compile=False)
            models['avoid'] = tf.keras.models.load_model(avoid_model_path, compile=False)
            print("Models loaded successfully without compilation")
        except Exception as e2:
            print(f"Failed to load models: {e2}")
            raise e2
    
    # Validate model input shapes
    target_input_shape = models['target'].input_shape
    avoid_input_shape = models['avoid'].input_shape
    
    print(f"Target model input shape: {target_input_shape}")
    print(f"Avoid model input shape: {avoid_input_shape}")
    
    # Check if input shapes are compatible
    if len(target_input_shape) != len(avoid_input_shape):
        raise ValueError(f"Model input shapes don't match: {target_input_shape} vs {avoid_input_shape}")
    
    # Expected shape: (batch_size, sequence_length, 4, 1) for one-hot encoded sequences
    expected_dims = 4
    if len(target_input_shape) != expected_dims:
        print(f"Warning: Expected input shape with {expected_dims} dimensions, got {len(target_input_shape)}")
        print("Make sure your sequences are properly one-hot encoded")
    
    return models

def predict_target_avoid_reward(seqs, models, batch_size=512):
    """
    Predict using target and avoid models and calculate reward.
    
    Args:
        seqs: Input sequences in one-hot format
        models: Dictionary with 'target' and 'avoid' models
        batch_size: Batch size for prediction
        
    Returns:
        Dictionary with predictions and reward
    """
    # Get predictions from both models
    target_preds = models['target'].predict(seqs, batch_size=batch_size)
    avoid_preds = models['avoid'].predict(seqs, batch_size=batch_size)
    
    # Handle different output shapes
    if len(target_preds.shape) > 1 and target_preds.shape[1] > 1:
        target_preds = target_preds[:, 0]  # Take first output for regression
    if len(avoid_preds.shape) > 1 and avoid_preds.shape[1] > 1:
        avoid_preds = avoid_preds[:, 0]  # Take first output for regression
    
    # Calculate reward
    reward = target_avoid_reward(target_preds, avoid_preds)
    
    return {
        'target_predictions': target_preds,
        'avoid_predictions': avoid_preds,
        'reward': reward
    }

# =============================================================================
# MODIFIED SATURATION MUTAGENESIS FOR TARGET/AVOID
# =============================================================================

def saturation_mutagenesis_target_avoid(seqs, models, iterations=10, 
                                       batch_size=128, mutation_batch_size=1024, return_intermediate_sequences=True):
    """
    Perform saturation mutagenesis using target vs avoid reward.
    
    Args:
        seqs: Input sequences in one-hot format
        models: Dictionary with 'target' and 'avoid' models
        iterations: Number of mutation iterations
        batch_size: Batch size for prediction
        return_intermediate_sequences: Whether to return intermediate sequences
        
    Returns:
        Tuple of (final_sequences, best_rewards, all_rewards, intermediate_sequences)
    """
    current_seqs = seqs.copy()
    all_rewards = []
    intermediate_sequences = []
    
    # Create progress bar for iterations
    iter_pbar = tqdm(range(iterations), desc="Saturation mutagenesis", unit="iter")
    
    for iteration in iter_pbar:
        iter_pbar.set_description(f"Saturation mutagenesis iteration {iteration + 1}/{iterations}")
        
        # Get current reward
        current_results = predict_target_avoid_reward(current_seqs, models, batch_size)
        current_reward = current_results['reward']
        all_rewards.append(current_reward.copy())
        
        if return_intermediate_sequences:
            intermediate_sequences.append(current_seqs.copy())
        
        # Create mutations in parallel batches for better GPU utilization
        print(f"  Creating mutations for {len(current_seqs)} sequences...")
        
        # Process each sequence individually (simpler and more reliable)
        mutated_seqs = []
        
        print(f"  Processing {len(current_seqs)} sequences...")
        
        for seq_idx, (seq, reward) in enumerate(tqdm(zip(current_seqs, current_reward), 
                                                   total=len(current_seqs), 
                                                   desc="  Processing sequences", 
                                                   unit="seq")):
            mutated_seq = create_systematic_mutations_target_avoid(
                seq, reward, models, mutation_batch_size
            )
            mutated_seqs.append(mutated_seq)
        
        mutated_seqs = np.array(mutated_seqs)
        
        # Evaluate mutations
        print(f"  Evaluating {len(mutated_seqs)} mutated sequences...")
        mutated_results = predict_target_avoid_reward(mutated_seqs, models, batch_size)
        mutated_reward = mutated_results['reward']
        
        # Keep better sequences (higher reward is better)
        better_mask = mutated_reward > current_reward
        improvements = np.sum(better_mask)
        
        # Ensure both arrays have the same shape for comparison
        if current_reward.shape != mutated_reward.shape:
            print(f"  Warning: Shape mismatch - current_reward: {current_reward.shape}, mutated_reward: {mutated_reward.shape}")
            # Flatten both to 1D for comparison
            current_reward_flat = current_reward.flatten()
            mutated_reward_flat = mutated_reward.flatten()
            better_mask = mutated_reward_flat > current_reward_flat
        else:
            better_mask = mutated_reward > current_reward
        
        # Ensure mask is 1D for proper indexing
        if better_mask.ndim > 1:
            better_mask = better_mask.flatten()
        
        current_seqs[better_mask] = mutated_seqs[better_mask]
        current_reward[better_mask] = mutated_reward[better_mask]
        
        print(f"  Best reward this iteration: {np.max(mutated_reward):.4f}")
        print(f"  Sequences improved: {improvements}/{len(mutated_reward)}")
        print(f"  Current best reward: {np.max(current_reward):.4f}")
        
        # Update iteration progress bar
        iter_pbar.set_postfix({
            'best_reward': f"{np.max(mutated_reward):.2e}",
            'improvements': f"{improvements}/{len(mutated_reward)}"
        })
    
    iter_pbar.close()
    
    # Final reward evaluation
    final_results = predict_target_avoid_reward(current_seqs, models, batch_size)
    final_reward = final_results['reward']
    all_rewards.append(final_reward)
    
    if return_intermediate_sequences:
        intermediate_sequences.append(current_seqs.copy())
    
    best_rewards = np.max(all_rewards, axis=0)
    
    return current_seqs, best_rewards, np.array(all_rewards), intermediate_sequences

def create_systematic_mutations_target_avoid(seq, current_reward, models, batch_size):
    """Create systematic mutations to find the best single-position mutation using optimized batch processing."""
    best_seq = seq.copy()
    best_reward = current_reward
    seq_len = seq.shape[0]
    
    # Collect all possible mutations
    all_mutations = []
    
    for pos in range(seq_len):
        for new_base_idx in range(4):
            # Skip if it's the same as current base
            if seq[pos, new_base_idx] == 1:
                continue
            
            # Create mutation
            mutated = seq.copy()
            mutated[pos, :] = 0
            mutated[pos, new_base_idx] = 1
            all_mutations.append(mutated)
    
    if not all_mutations:
        return best_seq
    
    # Process mutations in batches for GPU efficiency
    all_mutations = np.array(all_mutations)
    best_reward = current_reward
    
    # Use reasonable batch size to avoid memory issues
    effective_batch_size = min(batch_size, len(all_mutations))
    
    # Process in batches for GPU efficiency
    for i in tqdm(range(0, len(all_mutations), effective_batch_size), 
                  desc="    Testing mutations", unit="batch", leave=False):
        batch_end = min(i + effective_batch_size, len(all_mutations))
        batch_mutations = all_mutations[i:batch_end]
        
        # Evaluate batch with error handling
        try:
            results = predict_target_avoid_reward(batch_mutations, models, effective_batch_size)
            batch_rewards = results['reward']
        except tf.errors.ResourceExhaustedError:
            print(f"    Memory error with batch size {effective_batch_size}, reducing...")
            # Try with smaller batch size
            smaller_batch_size = effective_batch_size // 2
            if smaller_batch_size < 1:
                smaller_batch_size = 1
            results = predict_target_avoid_reward(batch_mutations, models, smaller_batch_size)
            batch_rewards = results['reward']
        
        # Find best in this batch
        best_in_batch_idx = np.argmax(batch_rewards)
        best_in_batch_reward = batch_rewards[best_in_batch_idx]
        
        # Update global best if this batch has a better mutation
        if best_in_batch_reward > best_reward:
            best_seq = batch_mutations[best_in_batch_idx]
            best_reward = best_in_batch_reward
    
    return best_seq

# =============================================================================
# MODIFIED ADALEAD FOR TARGET/AVOID
# =============================================================================

class adalead_target_avoid:
    """
    Adaptive Lead optimization for target vs avoid models.
    """
    
    def __init__(self, model_queries_per_batch, eval_batch_size, models):
        """
        Initialize adalead optimizer for target/avoid models.
        
        Args:
            model_queries_per_batch: Number of model queries per batch
            eval_batch_size: Batch size for evaluation
            models: Dictionary with 'target' and 'avoid' models
        """
        self.model_queries_per_batch = model_queries_per_batch
        self.eval_batch_size = eval_batch_size
        self.models = models
    
    def propose_sequences(self, current_sequences):
        """
        Propose new sequences based on current sequences.
        
        Args:
            current_sequences: Current set of sequences
            
        Returns:
            Tuple of (new_sequences, predicted_rewards)
        """
        n_seqs = current_sequences.shape[0]
        new_sequences = []
        
        # Generate new sequences through recombination and mutation
        for i in range(n_seqs):
            # Select parent sequences (including self)
            parent1 = current_sequences[i]
            parent2 = current_sequences[np.random.randint(0, n_seqs)]
            
            # Create recombination
            recombined = self._recombine_sequences(parent1, parent2)
            
            # Apply mutations
            mutated = self._mutate_sequence(recombined)
            
            new_sequences.append(mutated)
        
        new_sequences = np.array(new_sequences)
        
        # Evaluate reward
        results = predict_target_avoid_reward(new_sequences, self.models, self.eval_batch_size)
        predicted_rewards = results['reward']
        
        return new_sequences, predicted_rewards
    
    def _recombine_sequences(self, seq1, seq2):
        """Recombine two sequences."""
        if len(seq1.shape) == 4:
            seq1 = seq1.squeeze(axis=-1)
        if len(seq2.shape) == 4:
            seq2 = seq2.squeeze(axis=-1)
        
        # Random crossover point
        crossover_point = np.random.randint(1, seq1.shape[0])
        
        # Create recombined sequence
        recombined = seq1.copy()
        recombined[crossover_point:] = seq2[crossover_point:]
        
        return recombined
    
    def _mutate_sequence(self, seq):
        """Apply random mutations to a sequence."""
        if len(seq.shape) == 4:
            seq = seq.squeeze(axis=-1)
        
        mutated = seq.copy()
        seq_len = seq.shape[0]
        
        # Number of mutations (1-3)
        num_mutations = np.random.randint(1, 4)
        
        for _ in range(num_mutations):
            pos = np.random.randint(0, seq_len)
            new_base_idx = np.random.randint(0, 4)
            mutated[pos, :] = 0
            mutated[pos, new_base_idx] = 1
        
        return mutated

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Target vs Avoid Model Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mutate_sequences_target_avoid.py --target GLUT6 --avoid GLUT5
  python mutate_sequences_target_avoid.py --target GLUT7 --avoid EXC --iterations 50
  python mutate_sequences_target_avoid.py --target L2.3.IT --avoid L4.IT --n-seqs 50 --adalead-rounds 10
  python mutate_sequences_target_avoid.py --target L3.IT --avoid L4.IT --input-fasta sequences.fasta
        """
    )
    
    parser.add_argument('--target', required=True, 
                       help='Target cell type name (e.g., GLUT6, GLUT7, L2.3.IT)')
    parser.add_argument('--avoid', required=True,
                       help='Avoid cell type name (e.g., GLUT5, EXC, L4.IT)')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of saturation mutagenesis iterations (default: 20)')
    parser.add_argument('--seq-length', type=int, default=500,
                       help='Sequence length for generated sequences (default: 500)')
    parser.add_argument('--n-seqs', type=int, default=200,
                       help='Number of sequences to optimize (default: 200)')
    parser.add_argument('--adalead-rounds', type=int, default=50,
                       help='Number of AdaLead optimization rounds (default: 50)')
    parser.add_argument('--adalead-recombine-turns', type=int, default=4,
                       help='Number of AdaLead recombination turns per round (default: 4)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for model predictions (default: 256)')
    parser.add_argument('--mutation-batch-size', type=int, default=128,
                       help='Batch size for systematic mutations (default: 128, adjust based on GPU memory)')
    parser.add_argument('--output-prefix', default='target_avoid_optimization',
                       help='Prefix for output files (default: target_avoid_optimization)')
    parser.add_argument('--input-fasta', default=None,
                       help='Path to FASTA file to load initial sequences from (optional, defaults to random sequences)')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    print(f"Target cell type: {args.target}")
    print(f"Avoid cell type: {args.avoid}")
    print(f"Iterations: {args.iterations}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of sequences: {args.n_seqs}")
    print("=" * 50)
    
    # Check GPU memory and adjust batch sizes if needed
    gpu_memory = get_gpu_memory_info()
    if gpu_memory and gpu_memory < 8:  # Less than 8GB GPU memory
        print(f"Detected {gpu_memory:.1f}GB GPU memory, reducing batch sizes...")
        args.batch_size = min(args.batch_size, 128)
        args.mutation_batch_size = min(args.mutation_batch_size, 64)
        print(f"Adjusted batch sizes: prediction={args.batch_size}, mutation={args.mutation_batch_size}")
    
    # Initialize wandb
    print("Initializing wandb tracking...")
    wandb_run = init_wandb(args.target, args.avoid, args)
    if wandb_run is not None:
        print(f"Wandb run: mutation_script_{args.target}_vs_{args.avoid}")
    else:
        print("Running without wandb tracking...")
     
    # Get model paths
    try:
        target_model_path, avoid_model_path = get_model_paths(args.target, args.avoid)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load models
    print("Loading target and avoid models...")
    try:
        models = load_target_avoid_models(target_model_path, avoid_model_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading models: {e}")
        return
    
    # Load sequences from FASTA file or create random sequences
    target_input_shape = models['target'].input_shape
    if len(target_input_shape) == 3:  # (batch, seq_len, features)
        model_seq_length = target_input_shape[1]
    elif len(target_input_shape) == 4:  # (batch, seq_len, features, 1)
        model_seq_length = target_input_shape[1]
    else:
        model_seq_length = args.seq_length
        print(f"Warning: Could not determine sequence length from model shape {target_input_shape}, using {model_seq_length}")
    
    print(f"Model expects sequences of length: {model_seq_length}")
    
    if args.input_fasta:
        # Load sequences from FASTA file
        print(f"Loading sequences from FASTA file: {args.input_fasta}")
        if not os.path.exists(args.input_fasta):
            raise FileNotFoundError(f"FASTA file not found: {args.input_fasta}")
        
        sequences = seq_functions.get_fasta_seqs(args.input_fasta)
        
        # Check sequence lengths match model expectations
        if sequences.shape[1] != model_seq_length:
            print(f"Warning: Sequences in FASTA have length {sequences.shape[1]}, but model expects {model_seq_length}")
            print(f"Using sequences as-is. Consider filtering sequences by length if needed.")
        
        # Ensure sequences have the correct shape for the model
        if len(target_input_shape) == 3 and sequences.ndim == 4:
            # Model expects (N, L, 4) but sequences are (N, L, 4, 1)
            sequences = np.squeeze(sequences, axis=-1)
        elif len(target_input_shape) == 4 and sequences.ndim == 3:
            # Model expects (N, L, 4, 1) but sequences are (N, L, 4)
            sequences = np.expand_dims(sequences, axis=-1)
        
        # Limit number of sequences if more than requested
        if sequences.shape[0] > args.n_seqs:
            print(f"FASTA contains {sequences.shape[0]} sequences, limiting to {args.n_seqs}")
            sequences = sequences[:args.n_seqs]
        elif sequences.shape[0] < args.n_seqs:
            print(f"FASTA contains {sequences.shape[0]} sequences, which is less than requested {args.n_seqs}")
            print(f"Using all {sequences.shape[0]} sequences from FASTA")
        
        print(f"Loaded {sequences.shape[0]} sequences of length {sequences.shape[1]} from FASTA")
    else:
        # Create random sequences for demonstration
        print("Creating random example sequences...")
        
        # Create sequences with the correct shape for the model
        if len(target_input_shape) == 3:  # (batch, seq_len, features)
            sequences = np.random.random((args.n_seqs, model_seq_length, 4))
            # Normalize to one-hot
            sequences = (sequences == sequences.max(axis=2, keepdims=True)).astype(float)
        elif len(target_input_shape) == 4:  # (batch, seq_len, features, 1)
            sequences = np.random.random((args.n_seqs, model_seq_length, 4, 1))
            # Normalize to one-hot
            sequences = (sequences == sequences.max(axis=2, keepdims=True)).astype(float)
        else:
            # Fallback to original method
            sequences = np.random.random((args.n_seqs, model_seq_length, 4, 1))
            sequences = (sequences == sequences.max(axis=2, keepdims=True)).astype(float)
        
        print(f"Created {sequences.shape[0]} random sequences of length {sequences.shape[1]}")
    
    # Evaluate initial sequences
    print("Evaluating initial sequences...")
    initial_results = predict_target_avoid_reward(sequences, models, args.batch_size)
    print(f"Initial target predictions: {initial_results['target_predictions']}")
    print(f"Initial avoid predictions: {initial_results['avoid_predictions']}")
    print(f"Initial rewards: {initial_results['reward']}")
    
    # Log initial metrics to wandb
    log_reward_metrics(
        step=0, 
        reward_type='initial', 
        rewards=initial_results['reward'],
        target_preds=initial_results['target_predictions'],
        avoid_preds=initial_results['avoid_predictions']
    )
    
    # Run AdaLead optimization
    print(f"\nRunning AdaLead optimization ({args.adalead_rounds} rounds)...")
    adalead_obj = adalead_target_avoid(
        model_queries_per_batch=3*args.n_seqs + 1,  # Match original: 3*num_seed_seqs + 1
        eval_batch_size=args.n_seqs,
        models=models
    )
    
    # Run multiple rounds of AdaLead (matching original structure)
    current_seqs = sequences.copy()
    current_rewards = initial_results['reward'].copy()
    step_counter = 0
    
    # Create progress bar for AdaLead
    total_adalead_steps = args.adalead_rounds * args.adalead_recombine_turns
    adalead_pbar = tqdm(total=total_adalead_steps, desc="AdaLead optimization", unit="step")
    
    for round_num in range(args.adalead_rounds):
        # Run multiple recombination turns per round
        for turn_num in range(args.adalead_recombine_turns):
            step_counter += 1
            new_seqs, rewards = adalead_obj.propose_sequences(current_seqs)
            
            # Keep better sequences - compare with current best rewards
            better_mask = rewards.flatten() > current_rewards.flatten()
            improvements = np.sum(better_mask)
            current_seqs[better_mask] = new_seqs[better_mask]
            current_rewards[better_mask] = rewards[better_mask]
            
            # Update progress bar
            adalead_pbar.set_postfix({
                'best_reward': f"{np.max(rewards):.2e}",
                'improvements': f"{improvements}/{len(rewards)}"
            })
            adalead_pbar.update(1)
            
            # Log AdaLead metrics to wandb
            log_reward_metrics(
                step=step_counter,
                reward_type='adalead',
                rewards=rewards
            )
    
    adalead_pbar.close()
    print(f"AdaLead completed! Best reward achieved: {np.max(current_rewards):.4f}")
    
    # Run saturation mutagenesis
    print(f"\nRunning saturation mutagenesis ({args.iterations} iterations)...")
    final_seqs, best_rewards, all_rewards, intermed_seqs = saturation_mutagenesis_target_avoid(
        current_seqs, models, iterations=args.iterations, 
        batch_size=args.batch_size, mutation_batch_size=args.mutation_batch_size, 
        return_intermediate_sequences=True
    )
    
    # Log saturation mutagenesis metrics to wandb
    for iteration in range(args.iterations):
        step_counter += 1
        log_reward_metrics(
            step=step_counter,
            reward_type='saturation_mutagenesis',
            rewards=all_rewards[iteration]
        )
    
    # Final evaluation
    final_results = predict_target_avoid_reward(final_seqs, models, args.batch_size)
    print(f"\nFinal results:")
    print(f"Best target predictions: {np.max(final_results['target_predictions']):.4f}")
    print(f"Best avoid predictions: {np.max(final_results['avoid_predictions']):.4f}")
    print(f"Best rewards: {np.max(final_results['reward']):.4f}")
    
    # Log final metrics to wandb
    log_reward_metrics(
        step=step_counter + 1,
        reward_type='final',
        rewards=final_results['reward'],
        target_preds=final_results['target_predictions'],
        avoid_preds=final_results['avoid_predictions']
    )
    
    # Create summary table for wandb
    summary_data = []
    summary_data.append(['Metric', 'Initial', 'Final', 'Improvement'])
    summary_data.append(['Reward Mean', f"{np.mean(initial_results['reward']):.4f}", 
                        f"{np.mean(final_results['reward']):.4f}",
                        f"{np.mean(final_results['reward']) - np.mean(initial_results['reward']):.4f}"])
    summary_data.append(['Reward Max', f"{np.max(initial_results['reward']):.4f}", 
                        f"{np.max(final_results['reward']):.4f}",
                        f"{np.max(final_results['reward']) - np.max(initial_results['reward']):.4f}"])
    summary_data.append(['Target Pred Mean', f"{np.mean(initial_results['target_predictions']):.4f}", 
                        f"{np.mean(final_results['target_predictions']):.4f}",
                        f"{np.mean(final_results['target_predictions']) - np.mean(initial_results['target_predictions']):.4f}"])
    summary_data.append(['Avoid Pred Mean', f"{np.mean(initial_results['avoid_predictions']):.4f}", 
                        f"{np.mean(final_results['avoid_predictions']):.4f}",
                        f"{np.mean(final_results['avoid_predictions']) - np.mean(initial_results['avoid_predictions']):.4f}"])
    
    # Log summary table to wandb
    if wandb.run is not None:
        wandb.log({"summary_table": wandb.Table(data=summary_data[1:], columns=summary_data[0])})
    
    # Save results
    print("\nSaving results...")
    results_dict = {
        'target_cell_type': args.target,
        'avoid_cell_type': args.avoid,
        'initial_sequences': sequences,
        'final_sequences': final_seqs,
        'initial_results': initial_results,
        'final_results': final_results,
        'all_rewards': all_rewards,
        'intermediate_sequences': intermed_seqs,
        'parameters': {
            'iterations': args.iterations,
            'seq_length': args.seq_length,
            'n_seqs': args.n_seqs,
            'adalead_rounds': args.adalead_rounds,
            'batch_size': args.batch_size
        }
    }
    
    output_file = f"{args.output_prefix}_{args.target}_vs_{args.avoid}_results.pkl"
    with open(output_file, 'wb') as f:
        dill.dump(results_dict, f)
    
    print(f"Results saved to {output_file}")
    
    # Plot results
    print("Creating plots...")
    plt.figure(figsize=(12, 8))
    
    # Plot reward progression
    plt.subplot(2, 2, 1)
    plt.plot(np.max(all_rewards, axis=1))
    plt.title('Best Reward Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    
    # Plot target vs avoid predictions
    plt.subplot(2, 2, 2)
    plt.scatter(final_results['avoid_predictions'], final_results['target_predictions'], alpha=0.6)
    plt.xlabel('Avoid Model Predictions')
    plt.ylabel('Target Model Predictions')
    plt.title('Target vs Avoid Predictions')
    
    # Plot reward distribution
    plt.subplot(2, 2, 3)
    plt.hist(final_results['reward'], bins=20, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Final Reward Distribution')
    
    # Plot reward vs target prediction
    plt.subplot(2, 2, 4)
    plt.scatter(final_results['target_predictions'], final_results['reward'], alpha=0.6)
    plt.xlabel('Target Model Predictions')
    plt.ylabel('Reward')
    plt.title('Reward vs Target Predictions')
    
    plt.tight_layout()
    plot_file = f"{args.output_prefix}_{args.target}_vs_{args.avoid}_results.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {plot_file}")
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
        print("Wandb tracking completed!")
    print("Optimization complete!")

if __name__ == "__main__":
    main()