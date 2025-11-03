"""
Helper functions for synthetic sequence design and mutation analysis.
This module contains all the functions originally from synthESizer repository
that are needed for mutate_sequences.py to work independently.
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
import random
from typing import Dict, List, Tuple, Union, Optional
import warnings

# =============================================================================
# SEQUENCE FUNCTIONS (from seq_functions module)
# =============================================================================

def get_fasta_seqs(fasta_path: str) -> np.ndarray:
    """
    Load sequences from FASTA file and convert to one-hot encoding.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        numpy array of shape (n_sequences, sequence_length, 4, 1) in one-hot format
    """
    sequences = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq).upper()
        # Convert to one-hot encoding
        onehot = dna_to_onehot(seq_str)
        sequences.append(onehot)
    
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")
    
    return np.array(sequences)

def dna_to_onehot(seq: str) -> np.ndarray:
    """
    Convert DNA sequence string to one-hot encoding.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        numpy array of shape (sequence_length, 4) in one-hot format
    """
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    
    # Handle ambiguous bases by using random choice
    ambiguous_bases = {'N': ['A', 'C', 'G', 'T'], 'R': ['A', 'G'], 'Y': ['C', 'T'], 
                      'S': ['G', 'C'], 'W': ['A', 'T'], 'K': ['G', 'T'], 'M': ['A', 'C']}
    
    onehot_seq = []
    for base in seq:
        if base in mapping:
            onehot_seq.append(mapping[base])
        elif base in ambiguous_bases:
            # Randomly choose from ambiguous base options
            chosen_base = random.choice(ambiguous_bases[base])
            onehot_seq.append(mapping[chosen_base])
        else:
            # Default to N (random choice)
            chosen_base = random.choice(['A', 'C', 'G', 'T'])
            onehot_seq.append(mapping[chosen_base])
    
    return np.array(onehot_seq)

def onehot_to_dna(onehot: np.ndarray) -> str:
    """
    Convert one-hot encoded sequence to DNA string.
    
    Args:
        onehot: numpy array of shape (sequence_length, 4) in one-hot format
        
    Returns:
        DNA sequence string
    """
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    dna_seq = []
    for pos in onehot:
        base_idx = np.argmax(pos)
        dna_seq.append(mapping[base_idx])
    
    return ''.join(dna_seq)

def batch_inverse_onehot(seqs: np.ndarray) -> List[str]:
    """
    Convert batch of one-hot encoded sequences to DNA strings.
    
    Args:
        seqs: numpy array of shape (n_sequences, sequence_length, 4, 1) or (n_sequences, sequence_length, 4)
        
    Returns:
        List of DNA sequence strings
    """
    if len(seqs.shape) == 4:
        # Remove the last dimension if present
        seqs = seqs.squeeze(axis=-1)
    
    dna_sequences = []
    for i in range(seqs.shape[0]):
        dna_seq = onehot_to_dna(seqs[i])
        dna_sequences.append(dna_seq)
    
    return dna_sequences

def write_fasta(seq_dict: Dict[str, str], output_path: str):
    """
    Write sequences to FASTA file.
    
    Args:
        seq_dict: Dictionary mapping sequence IDs to DNA sequences
        output_path: Path to output FASTA file
    """
    records = []
    for seq_id, sequence in seq_dict.items():
        record = SeqRecord(Seq(sequence), id=seq_id, description="")
        records.append(record)
    
    SeqIO.write(records, output_path, "fasta")

def compute_gc_content(seqs: np.ndarray) -> np.ndarray:
    """
    Compute GC content for each sequence in the batch.
    
    Args:
        seqs: numpy array of shape (n_sequences, sequence_length, 4, 1) in one-hot format
        
    Returns:
        numpy array of GC content values for each sequence
    """
    if len(seqs.shape) == 4:
        seqs = seqs.squeeze(axis=-1)
    
    gc_contents = []
    for i in range(seqs.shape[0]):
        seq = seqs[i]
        # G is index 2, C is index 1
        gc_count = np.sum(seq[:, 1]) + np.sum(seq[:, 2])  # C + G
        total_count = seq.shape[0]
        gc_content = gc_count / total_count if total_count > 0 else 0
        gc_contents.append(gc_content)
    
    return np.array(gc_contents)

def avg_pairwise_hamming_onehot(seqs: np.ndarray) -> float:
    """
    Compute average pairwise Hamming distance between sequences.
    
    Args:
        seqs: numpy array of shape (n_sequences, sequence_length, 4, 1) in one-hot format
        
    Returns:
        Average pairwise Hamming distance
    """
    if len(seqs.shape) == 4:
        seqs = seqs.squeeze(axis=-1)
    
    if seqs.shape[0] < 2:
        return 0.0
    
    # Convert to integer representation for distance calculation
    int_seqs = np.argmax(seqs, axis=2)  # Shape: (n_sequences, sequence_length)
    
    # Compute pairwise Hamming distances
    distances = pairwise_distances(int_seqs, metric='hamming') * seqs.shape[1]
    
    # Get upper triangle (excluding diagonal)
    upper_tri = np.triu(distances, k=1)
    non_zero = upper_tri[upper_tri > 0]
    
    return np.mean(non_zero) if len(non_zero) > 0 else 0.0

def max_pairwise_hamming_onehot(seqs: np.ndarray) -> np.ndarray:
    """
    Find the pair of sequences with maximum Hamming distance.
    
    Args:
        seqs: numpy array of shape (n_sequences, sequence_length, 4, 1) in one-hot format
        
    Returns:
        numpy array containing the two most different sequences
    """
    if len(seqs.shape) == 4:
        seqs = seqs.squeeze(axis=-1)
    
    if seqs.shape[0] < 2:
        return seqs
    
    # Convert to integer representation
    int_seqs = np.argmax(seqs, axis=2)
    
    # Compute pairwise Hamming distances
    distances = pairwise_distances(int_seqs, metric='hamming') * seqs.shape[1]
    
    # Find indices of maximum distance
    max_idx = np.unravel_index(np.argmax(distances), distances.shape)
    
    return seqs[list(max_idx)]

def mutate_onehot(seqs: np.ndarray, target_bases: List[str], new_bases: List[str], 
                 num_replacements: int) -> np.ndarray:
    """
    Mutate sequences by replacing target bases with new bases.
    
    Args:
        seqs: numpy array of shape (n_sequences, sequence_length, 4) or (n_sequences, sequence_length, 4, 1) in one-hot format
        target_bases: List of bases to replace (e.g., ['A', 'T'])
        new_bases: List of replacement bases (e.g., ['G', 'C'])
        num_replacements: Number of replacements to make per sequence
        
    Returns:
        Mutated sequences
    """
    if len(seqs.shape) == 4:
        seqs = seqs.squeeze(axis=-1)
    
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    new_bases_idx = [base_to_idx[base] for base in new_bases]
    
    mutated_seqs = seqs.copy()
    
    for i in range(seqs.shape[0]):
        seq = seqs[i]
        
        # Find positions with target bases
        target_positions = []
        for target_base in target_bases:
            target_idx = base_to_idx[target_base]
            positions = np.where(seq[:, target_idx] == 1)[0]
            target_positions.extend(positions)
        
        if len(target_positions) == 0:
            continue
        
        # Randomly select positions to mutate
        if num_replacements > len(target_positions):
            num_replacements = len(target_positions)
        
        selected_positions = random.sample(target_positions, num_replacements)
        
        # Perform mutations
        for pos in selected_positions:
            # Set all bases to 0
            mutated_seqs[i, pos, :] = 0
            # Set new base to 1
            new_base_idx = random.choice(new_bases_idx)
            mutated_seqs[i, pos, new_base_idx] = 1
    
    return mutated_seqs

# =============================================================================
# SATURATION MUTAGENESIS FUNCTIONS
# =============================================================================

def opt_models_predict(seqs: np.ndarray, opt_models: Dict, mean_type: str = 'arithmetic_min', 
                      alpha: float = 0.5, weights: Optional[np.ndarray] = None, 
                      zmeans: Optional[Dict] = None, zstds: Optional[Dict] = None, 
                      predict_batch_size: int = 512, flip_positives: bool = False) -> np.ndarray:
    """
    Make predictions using ensemble of optimized models.
    
    Args:
        seqs: Input sequences in one-hot format
        opt_models: Dictionary of trained models
        mean_type: Type of ensemble averaging ('arithmetic_min', 'geometric_mean', etc.)
        alpha: Weighting parameter for ensemble
        weights: Optional weights for models
        zmeans: Optional z-score means for normalization
        zstds: Optional z-score standard deviations for normalization
        predict_batch_size: Batch size for prediction
        flip_positives: Whether to flip positive predictions
        
    Returns:
        Ensemble predictions
    """
    predictions = []
    
    for name, model in opt_models.items():
        pred = model.predict(seqs, batch_size=predict_batch_size)
        
        # Handle different output shapes
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            pred = pred[:, 0]  # Take first output for regression
        
        # Apply z-score normalization if provided
        if zmeans is not None and zstds is not None and name in zmeans:
            pred = (pred - zmeans[name]) / zstds[name]
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Apply ensemble method
    if mean_type == 'arithmetic_min':
        # Use minimum of predictions (conservative approach)
        ensemble_pred = np.min(predictions, axis=0)
    elif mean_type == 'geometric_mean':
        # Geometric mean
        ensemble_pred = np.exp(np.mean(np.log(predictions + 1e-8), axis=0))
    elif mean_type == 'arithmetic_mean':
        # Simple arithmetic mean
        ensemble_pred = np.mean(predictions, axis=0)
    else:
        # Default to arithmetic mean
        ensemble_pred = np.mean(predictions, axis=0)
    
    # Apply weights if provided
    if weights is not None:
        ensemble_pred = ensemble_pred * weights
    
    # Flip positives if requested
    if flip_positives:
        ensemble_pred = -ensemble_pred
    
    return ensemble_pred

def saturation_mutagenesis_loop(seqs: np.ndarray, opt_models: Dict, iterations: int,
                               val_models: Optional[Dict] = None, mean_type: str = 'arithmetic_min',
                               alpha: float = 0.5, predict_batch_size: int = 128,
                               random_mode: bool = False, return_intermediate_sequences: bool = True,
                               keep_parent: bool = True, weights: Optional[np.ndarray] = None,
                               zmeans: Optional[Dict] = None, zstds: Optional[Dict] = None,
                               flip_positives: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, List]:
    """
    Perform saturation mutagenesis on sequences.
    
    Args:
        seqs: Input sequences in one-hot format
        opt_models: Dictionary of optimization models
        iterations: Number of mutation iterations
        val_models: Optional validation models
        mean_type: Type of ensemble averaging
        alpha: Weighting parameter
        predict_batch_size: Batch size for prediction
        random_mode: Whether to use random mutations
        return_intermediate_sequences: Whether to return intermediate sequences
        keep_parent: Whether to keep parent sequences
        weights: Optional model weights
        zmeans: Optional z-score means
        zstds: Optional z-score standard deviations
        flip_positives: Whether to flip positive predictions
        
    Returns:
        Tuple of (final_sequences, best_fitness_values, all_fitness_values, 
                 validation_fitnesses, intermediate_sequences)
    """
    current_seqs = seqs.copy()
    all_fitness_values = []
    intermediate_sequences = []
    val_fitnesses_by_iter = {}
    
    for iteration in range(iterations):
        print(f"Saturation mutagenesis iteration {iteration + 1}/{iterations}")
        
        # Get current fitness
        current_fitness = opt_models_predict(
            current_seqs, opt_models, mean_type, alpha, weights, 
            zmeans, zstds, predict_batch_size, flip_positives
        )
        
        all_fitness_values.append(current_fitness.copy())
        
        if return_intermediate_sequences:
            intermediate_sequences.append(current_seqs.copy())
        
        # Store validation fitnesses if validation models provided
        if val_models is not None:
            val_fitnesses = {}
            for val_name, val_model in val_models.items():
                val_pred = val_model.predict(current_seqs, batch_size=predict_batch_size)
                if len(val_pred.shape) > 1 and val_pred.shape[1] > 1:
                    val_pred = val_pred[:, 0]
                val_fitnesses[val_name] = val_pred
            val_fitnesses_by_iter[iteration] = val_fitnesses
        
        # Create mutations
        mutated_seqs = []
        for i, seq in enumerate(current_seqs):
            if random_mode:
                # Random mutations
                mutated_seq = create_random_mutations(seq)
            else:
                # Systematic mutations (one position at a time)
                mutated_seq = create_systematic_mutations(seq, current_fitness[i], opt_models, 
                                                        mean_type, alpha, weights, zmeans, zstds,
                                                        predict_batch_size, flip_positives)
            
            mutated_seqs.append(mutated_seq)
        
        mutated_seqs = np.array(mutated_seqs)
        
        # Evaluate mutations
        mutated_fitness = opt_models_predict(
            mutated_seqs, opt_models, mean_type, alpha, weights,
            zmeans, zstds, predict_batch_size, flip_positives
        )
        
        # Keep better sequences
        if keep_parent:
            # Keep parent if it's better
            better_mask = current_fitness >= mutated_fitness
            current_seqs[better_mask] = current_seqs[better_mask]
            current_seqs[~better_mask] = mutated_seqs[~better_mask]
        else:
            # Always use mutated sequences
            current_seqs = mutated_seqs
    
    # Final fitness evaluation
    final_fitness = opt_models_predict(
        current_seqs, opt_models, mean_type, alpha, weights,
        zmeans, zstds, predict_batch_size, flip_positives
    )
    
    all_fitness_values.append(final_fitness)
    if return_intermediate_sequences:
        intermediate_sequences.append(current_seqs.copy())
    
    best_fitness_values = np.max(all_fitness_values, axis=0)
    
    return current_seqs, best_fitness_values, np.array(all_fitness_values), val_fitnesses_by_iter, intermediate_sequences

def create_random_mutations(seq: np.ndarray) -> np.ndarray:
    """Create random mutations in a sequence."""
    mutated = seq.copy()
    seq_len = seq.shape[0]
    
    # Randomly select a position to mutate
    pos = np.random.randint(0, seq_len)
    
    # Randomly select a new base
    new_base_idx = np.random.randint(0, 4)
    
    # Set all bases to 0 at this position
    mutated[pos, :] = 0
    # Set new base to 1
    mutated[pos, new_base_idx] = 1
    
    return mutated

def create_systematic_mutations(seq: np.ndarray, current_fitness: float, opt_models: Dict,
                               mean_type: str, alpha: float, weights: Optional[np.ndarray],
                               zmeans: Optional[Dict], zstds: Optional[Dict],
                               predict_batch_size: int, flip_positives: bool) -> np.ndarray:
    """Create systematic mutations to find the best single-position mutation."""
    best_seq = seq.copy()
    best_fitness = current_fitness
    seq_len = seq.shape[0]
    
    # Try mutating each position
    for pos in range(seq_len):
        for new_base_idx in range(4):
            # Skip if it's the same as current base
            if seq[pos, new_base_idx] == 1:
                continue
            
            # Create mutation
            mutated = seq.copy()
            mutated[pos, :] = 0
            mutated[pos, new_base_idx] = 1
            
            # Evaluate fitness
            fitness = opt_models_predict(
                mutated[np.newaxis, :], opt_models, mean_type, alpha, weights,
                zmeans, zstds, predict_batch_size, flip_positives
            )[0]
            
            # Keep if better
            if fitness > best_fitness:
                best_seq = mutated
                best_fitness = fitness
    
    return best_seq

def predict_with_random_mutations(model, sequences: np.ndarray, task: str = 'regression',
                                 num_mutations: int = 50, num_trials: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with random mutations to estimate uncertainty.
    
    Args:
        model: Trained model
        sequences: Input sequences
        task: Task type ('regression' or 'classification')
        num_mutations: Number of mutations per trial
        num_trials: Number of trials
        
    Returns:
        Tuple of (mean_predictions, std_predictions)
    """
    all_predictions = []
    
    for trial in range(num_trials):
        # Create random mutations
        mutated_seqs = []
        for seq in sequences:
            mutated = seq.copy()
            for _ in range(num_mutations):
                pos = np.random.randint(0, seq.shape[0])
                new_base_idx = np.random.randint(0, 4)
                mutated[pos, :] = 0
                mutated[pos, new_base_idx] = 1
            mutated_seqs.append(mutated)
        
        mutated_seqs = np.array(mutated_seqs)
        
        # Make predictions
        pred = model.predict(mutated_seqs, batch_size=512)
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            if task == 'classification':
                pred = pred[:, 1]  # Take positive class probability
            else:
                pred = pred[:, 0]  # Take first output for regression
        
        all_predictions.append(pred)
    
    all_predictions = np.array(all_predictions)
    
    # Compute mean and standard deviation
    mean_preds = np.mean(all_predictions, axis=0)
    std_preds = np.std(all_predictions, axis=0)
    
    return mean_preds, std_preds

# =============================================================================
# ADALEAD FUNCTIONS
# =============================================================================

class adalead_onehot:
    """
    Adaptive Lead optimization for one-hot encoded sequences.
    This is a simplified implementation of the adalead algorithm.
    """
    
    def __init__(self, model_queries_per_batch: int, eval_batch_size: int,
                 opt_models: Dict, mean_type: str = 'arithmetic_min', alpha: float = 0.5,
                 weights: Optional[np.ndarray] = None, zmeans: Optional[Dict] = None,
                 zstds: Optional[Dict] = None, flip_positives: bool = False):
        """
        Initialize adalead optimizer.
        
        Args:
            model_queries_per_batch: Number of model queries per batch
            eval_batch_size: Batch size for evaluation
            opt_models: Dictionary of optimization models
            mean_type: Type of ensemble averaging
            alpha: Weighting parameter
            weights: Optional model weights
            zmeans: Optional z-score means
            zstds: Optional z-score standard deviations
            flip_positives: Whether to flip positive predictions
        """
        self.model_queries_per_batch = model_queries_per_batch
        self.eval_batch_size = eval_batch_size
        self.opt_models = opt_models
        self.mean_type = mean_type
        self.alpha = alpha
        self.weights = weights
        self.zmeans = zmeans
        self.zstds = zstds
        self.flip_positives = flip_positives
    
    def propose_sequences(self, current_sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propose new sequences based on current sequences.
        
        Args:
            current_sequences: Current set of sequences
            
        Returns:
            Tuple of (new_sequences, predicted_fitnesses)
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
        
        # Evaluate fitness
        predicted_fitnesses = opt_models_predict(
            new_sequences, self.opt_models, self.mean_type, self.alpha,
            self.weights, self.zmeans, self.zstds, self.eval_batch_size,
            self.flip_positives
        )
        
        return new_sequences, predicted_fitnesses
    
    def _recombine_sequences(self, seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
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
    
    def _mutate_sequence(self, seq: np.ndarray) -> np.ndarray:
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
# UTILITY FUNCTIONS
# =============================================================================

def tile_motif_random(seq: np.ndarray, motif: str, n: int) -> List[np.ndarray]:
    """
    Randomly tile a motif into a sequence.
    
    Args:
        seq: Input sequence in one-hot format
        motif: Motif to tile (e.g., 'TATA', 'ATATACA')
        n: Number of times to tile the motif
        
    Returns:
        List of sequences with tiled motifs
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = seq.shape[1] if len(seq.shape) == 4 else seq.shape[0]
    
    start = seq_len // 4
    end = 3 * seq_len // 4
    motif = motif.upper()
    inds = [mapping[c] for c in motif]
    m = len(inds)
    slots = np.arange(start, end - (m - 1), m)
    
    eye4 = np.eye(4, dtype=seq.dtype)
    result = [seq]
    
    for _ in range(n):
        if len(seq.shape) == 4:
            arr = result[-1][0, :, :, 0].copy()
        else:
            arr = result[-1].copy()
        
        pos = np.random.choice(slots)
        for i, b in enumerate(inds):
            if pos + i < arr.shape[0]:
                arr[pos + i] = eye4[b]
        
        if len(seq.shape) == 4:
            result.append(arr[np.newaxis, :, :, np.newaxis])
        else:
            result.append(arr[np.newaxis, :, :])
    
    return result