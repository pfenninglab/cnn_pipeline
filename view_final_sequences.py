#!/usr/bin/env python3
import argparse
import os
import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sys

BASES = np.array(["A", "C", "G", "T"])
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}

# Add current directory to path for local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# MODEL PATH DICTIONARY (same as mutate_sequences_target_avoid.py)
# =============================================================================

# Dictionary mapping cell type combinations to model paths
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
# MODEL LOADING (same as mutate_sequences_target_avoid.py)
# =============================================================================

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
    
    return models

# =============================================================================
# PREDICTION FUNCTIONS (same as mutate_sequences_target_avoid.py)
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
    # Ensure sequences have the right shape for the model
    # Models might expect (N, L, 4, 1) instead of (N, L, 4)
    if seqs.ndim == 3:
        # Check model input shape to determine if we need to add a dimension
        target_input_shape = models['target'].input_shape
        if len(target_input_shape) == 4 and target_input_shape[-1] == 1:
            # Add trailing dimension
            seqs = np.expand_dims(seqs, axis=-1)
    
    # Get predictions from both models
    target_preds = models['target'].predict(seqs, batch_size=batch_size, verbose=0)
    avoid_preds = models['avoid'].predict(seqs, batch_size=batch_size, verbose=0)
    
    # Handle different output shapes
    if len(target_preds.shape) > 1 and target_preds.shape[1] > 1:
        target_preds = target_preds[:, 0]  # Take first output for regression
    if len(avoid_preds.shape) > 1 and avoid_preds.shape[1] > 1:
        avoid_preds = avoid_preds[:, 0]  # Take first output for regression
    
    # Flatten to 1D if needed
    target_preds = target_preds.flatten()
    avoid_preds = avoid_preds.flatten()
    
    # Calculate reward
    reward = target_avoid_reward(target_preds, avoid_preds)
    
    return {
        'target_predictions': target_preds,
        'avoid_predictions': avoid_preds,
        'reward': reward
    }


def load_final_sequences(pkl_path: str) -> np.ndarray:
    """Load final one-hot sequences from a results pickle.

    Returns array shaped (N, L, 4) (will squeeze a trailing channel if present).
    """
    with open(pkl_path, "rb") as f:
        obj = dill.load(f)
    if "final_sequences" not in obj:
        raise KeyError("final_sequences not found in pickle. Available keys: " + ", ".join(obj.keys()))
    seqs = obj["final_sequences"]
    seqs = np.asarray(seqs)
    # Handle shapes (N, L, 4) or (N, L, 4, 1)
    if seqs.ndim == 4 and seqs.shape[-1] == 1:
        seqs = np.squeeze(seqs, axis=-1)
    if seqs.ndim != 3 or seqs.shape[-1] != 4:
        raise ValueError(f"Unexpected sequences shape {seqs.shape}; expected (N, L, 4)[,1]")
    return seqs


def onehot_to_strings(onehot: np.ndarray) -> list:
    """Convert one-hot sequences (N, L, 4) to list of DNA strings."""
    idx = onehot.argmax(axis=-1)  # (N, L)
    return ["".join(BASES[i] for i in row) for row in idx]


def write_fasta(strings: list, fasta_path: str, prefix: str = "seq") -> None:
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(strings):
            f.write(f">{prefix}_{i}\n{seq}\n")


def compute_position_frequencies(onehot: np.ndarray) -> np.ndarray:
    """Compute nucleotide frequency per position.

    Returns array (L, 4) with frequencies summing to 1 across axis=1.
    """
    N, L, four = onehot.shape
    assert four == 4
    # Sum one-hot across sequences => counts per position per base
    counts = onehot.sum(axis=0)  # (L, 4)
    totals = counts.sum(axis=1, keepdims=True) + 1e-12  # avoid div-by-zero
    freqs = counts / totals
    return freqs


def compute_gc_content(onehot: np.ndarray) -> float:
    """Compute average GC content across all sequences as a percentage.
    
    Args:
        onehot: Array shaped (N, L, 4) with one-hot encoded sequences
        
    Returns:
        GC content as a percentage (0-100)
    """
    N, L, four = onehot.shape
    assert four == 4
    
    # G is at index 2, C is at index 1 in BASES array ["A", "C", "G", "T"]
    gc_counts = onehot[:, :, 1].sum() + onehot[:, :, 2].sum()  # C + G
    total_bases = N * L
    
    gc_content = (gc_counts / total_bases) * 100.0
    return gc_content

def compute_gc_content_per_sequence(onehot: np.ndarray) -> np.ndarray:
    """Compute GC content for each sequence as a percentage.
    
    Args:
        onehot: Array shaped (N, L, 4) with one-hot encoded sequences
        
    Returns:
        Array of GC content percentages (N,) with values 0-100
    """
    N, L, four = onehot.shape
    assert four == 4
    
    # G is at index 2, C is at index 1 in BASES array ["A", "C", "G", "T"]
    # Sum C and G counts per sequence: (N, L) -> (N,)
    gc_counts = onehot[:, :, 1].sum(axis=1) + onehot[:, :, 2].sum(axis=1)  # (N,)
    total_bases = L
    
    gc_content = (gc_counts / total_bases) * 100.0
    return gc_content


def plot_max_letter_per_position(freqs: np.ndarray, out_path: str, title: str = None, dpi: int = 200):
    """Plot a simple logo-like track: one letter per position sized by its frequency,
    with subplots showing individual nucleotide frequencies.

    freqs: (L, 4) frequencies for A,C,G,T.
    """
    L = freqs.shape[0]
    # Choose max base per position and its freq
    max_idx = freqs.argmax(axis=1)
    max_freq = freqs[np.arange(L), max_idx]
    letters = BASES[max_idx]

    # Basic styling for bases (colors)
    base_color = {
        "A": "#1f77b4",  # blue
        "C": "#ff7f0e",  # orange
        "G": "#2ca02c",  # green
        "T": "#d62728",  # red
    }

    # Create figure with subplots: main plot on top, 4 nucleotide plots below
    fig_width = max(10, L / 25)
    fig = plt.figure(figsize=(fig_width, 8))
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 1, 1, 1, 1], hspace=0.4)
    
    # Main plot (dominant nucleotide)
    ax_main = fig.add_subplot(gs[0])
    ax_main.set_xlim(0, L)
    ax_main.set_ylim(0, 1.15)  # Increased top margin to 1.15 from 1.05
    ax_main.set_xlabel("Position")
    ax_main.set_ylabel("Frequency of max nucleotide")
    if title:
        ax_main.set_title(title)
    else:
        ax_main.set_title("Dominant nucleotide frequency")

    # Draw a faint baseline
    ax_main.axhline(0, color="#cccccc", linewidth=0.8)

    # Plot letters with controlled sizing to prevent overlap
    # Use square root scaling for font size to prevent excessive growth
    base_font = 8
    max_font = 20  # Cap maximum font size
    for x in range(L):
        letter = letters[x]
        freq = float(max_freq[x])
        color = base_color[letter]
        
        # Scale font size using square root to prevent overlap
        # This makes letters grow more slowly at high frequencies
        fontsize = base_font + (max_font - base_font) * np.sqrt(freq)
        
        # Position letter at the top of the bar (at the frequency value)
        # This prevents vertical overlap while showing the frequency visually
        y_pos = freq
        
        # Use clip_on to ensure letters don't extend beyond their position bin
        ax_main.text(x + 0.5, y_pos, letter, ha="center", va="bottom", 
                fontsize=fontsize, color=color, family="DejaVu Sans Mono",
                clip_on=True)
        
        # Draw a vertical bar to show frequency height
        ax_main.add_patch(plt.Rectangle((x + 0.25, 0), 0.5, freq, color=color, alpha=0.15, lw=0))

    # Ticks every ~50bp for long sequences
    if L > 60:
        step = 50
        ax_main.set_xticks(np.arange(0, L + 1, step))
    else:
        ax_main.set_xticks(np.arange(0, L, 5))

    # Create 4 subplots for individual nucleotides
    base_names = ["A", "C", "G", "T"]
    base_indices = [0, 1, 2, 3]
    
    for i, (base_name, base_idx) in enumerate(zip(base_names, base_indices)):
        ax = fig.add_subplot(gs[i + 1])
        color = base_color[base_name]
        
        # Extract frequency for this nucleotide
        nuc_freq = freqs[:, base_idx]
        
        # Plot line
        ax.plot(range(L), nuc_freq, color=color, linewidth=1.5, label=base_name)
        ax.fill_between(range(L), 0, nuc_freq, color=color, alpha=0.2)
        
        ax.set_xlim(0, L)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Position" if i == 3 else "")  # Only bottom subplot gets xlabel
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis label
        ax.set_ylabel(f"{base_name}", rotation=0, labelpad=10, fontsize=10)
        
        # Set ticks
        if L > 60:
            step = 50
            ax.set_xticks(np.arange(0, L + 1, step))
            if i < 3:  # Hide x-axis labels for top 3 subplots
                ax.set_xticklabels([])
        else:
            ax.set_xticks(np.arange(0, L, 5))
            if i < 3:
                ax.set_xticklabels([])

    # Use subplots_adjust instead of tight_layout for GridSpec layouts
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.4)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="View final sequences and plot per-position dominant nucleotide frequency.")
    parser.add_argument("--input-pkl", required=True, help="Path to pickle file containing final_sequences")
    parser.add_argument("--output-fasta", required=False, default=None, help="Path to write sequences as FASTA")
    parser.add_argument("--output-plot", required=False, default=None, help="Path to write frequency plot (png/pdf)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences exported to FASTA (optional)")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--target", required=False, default=None, help="Target cell type name (required for predictions)")
    parser.add_argument("--avoid", required=False, default=None, help="Avoid cell type name (required for predictions)")
    parser.add_argument("--output-csv", required=False, default=None, help="Path to write CSV with predictions")
    parser.add_argument("--output-prediction-plot", required=False, default=None, help="Path to write prediction plots (scatter and histogram)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for model predictions (default: 512)")
    args = parser.parse_args()

    onehot = load_final_sequences(args.input_pkl)

    # Determine if we need predictions (for CSV, plots, or limiting by reward)
    need_predictions = (args.output_csv is not None or 
                       args.output_prediction_plot is not None or 
                       (args.limit is not None and (args.target is not None or args.avoid is not None)))
    
    # Load models and get predictions if needed
    models = None
    results = None
    df = None
    df_all = None  # Store full dataframe for plotting
    if need_predictions:
        if not args.target or not args.avoid:
            raise ValueError("--target and --avoid are required when using --output-csv, --output-prediction-plot, or --limit with predictions")
        
        print(f"\nLoading models for target={args.target}, avoid={args.avoid}...")
        try:
            target_model_path, avoid_model_path = get_model_paths(args.target, args.avoid)
            models = load_target_avoid_models(target_model_path, avoid_model_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading models: {e}")
            return
        
        print(f"Getting predictions for {len(onehot)} sequences...")
        results = predict_target_avoid_reward(onehot, models, batch_size=args.batch_size)
        
        # Convert sequences to strings
        sequences = onehot_to_strings(onehot)
        
        # Calculate GC content for each sequence
        gc_contents = compute_gc_content_per_sequence(onehot)
        
        # Create DataFrame with all sequences
        df_all = pd.DataFrame({
            'sequence': sequences,
            'target_prediction': results['target_predictions'],
            'avoid_prediction': results['avoid_predictions'],
            'reward_prediction': results['reward'],
            'gc_content': gc_contents
        })
        
        # If limit is specified, select top N sequences by reward value
        if args.limit is not None:
            df = df_all.nlargest(args.limit, 'reward_prediction')
            # Update onehot to match filtered sequences
            # Get indices of top sequences
            top_indices = df.index.values
            onehot = onehot[top_indices]
            print(f"Selected top {args.limit} sequences by reward value")
        else:
            df = df_all

    # Calculate and display GC content
    gc_content = compute_gc_content(onehot)
    print(f"Average GC content: {gc_content:.2f}%")

    # Export FASTA if requested
    if args.output_fasta:
        strings = onehot_to_strings(onehot)
        # Limit already applied if predictions were used
        if args.limit is not None and not need_predictions:
            strings = strings[: args.limit]
        fasta_dir = os.path.dirname(os.path.abspath(args.output_fasta))
        if fasta_dir:
            os.makedirs(fasta_dir, exist_ok=True)
        write_fasta(strings, args.output_fasta)
        print(f"Wrote {len(strings)} sequences to {args.output_fasta}")

    # Plot per-position dominant nucleotide frequency if requested
    if args.output_plot:
        freqs = compute_position_frequencies(onehot)
        plot_dir = os.path.dirname(os.path.abspath(args.output_plot))
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        plot_max_letter_per_position(freqs, args.output_plot, title=args.title)
        print(f"Wrote frequency plot to {args.output_plot}")

    # Generate prediction plots if requested
    if args.output_prediction_plot:
        if results is None:
            raise ValueError("Predictions not available. --target and --avoid must be provided.")
        
        # Use df_all for first two plots (all sequences)
        df_plot = df_all if df_all is not None else df
        # Use df (filtered sequences) for GC vs reward plot to match CSV
        df_csv_plot = df if df is not None else df_plot
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Scatter plot: target vs avoid predictions (all sequences)
        ax1.scatter(df_plot['avoid_prediction'], df_plot['target_prediction'], alpha=0.6, s=20)
        ax1.set_xlabel('Avoid Model Predictions')
        ax1.set_ylabel('Target Model Predictions')
        ax1.set_title('Target vs Avoid Predictions')
        ax1.grid(True, alpha=0.3)
        
        # Histogram: reward values (all sequences)
        ax2.hist(df_plot['reward_prediction'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Reward Prediction')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Scatter plot: GC content vs reward (sequences in CSV, respects limit)
        ax3.scatter(df_csv_plot['gc_content'], df_csv_plot['reward_prediction'], alpha=0.6, s=20)
        ax3.set_xlabel('GC Content (%)')
        ax3.set_ylabel('Reward Prediction')
        ax3.set_title('GC Content vs Reward')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_dir = os.path.dirname(os.path.abspath(args.output_prediction_plot))
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(args.output_prediction_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Wrote prediction plots to {args.output_prediction_plot}")

    # Write CSV if requested
    if args.output_csv:
        if df is None:
            raise ValueError("Predictions not available. --target and --avoid must be provided.")
        
        csv_dir = os.path.dirname(os.path.abspath(args.output_csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote predictions CSV to {args.output_csv}")
        print(f"Summary statistics:")
        print(f"  Target predictions: mean={df['target_prediction'].mean():.4f}, std={df['target_prediction'].std():.4f}")
        print(f"  Avoid predictions: mean={df['avoid_prediction'].mean():.4f}, std={df['avoid_prediction'].std():.4f}")
        print(f"  Reward predictions: mean={df['reward_prediction'].mean():.4f}, std={df['reward_prediction'].std():.4f}")
        print(f"  GC content: mean={df['gc_content'].mean():.2f}%, std={df['gc_content'].std():.2f}%")


if __name__ == "__main__":
    main()
