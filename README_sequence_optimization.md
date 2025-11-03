# Sequence Optimization Workflow

This guide explains how to run the sequence optimization pipeline using `mutate_sequences_target_avoid.py` and then analyze results with `view_final_sequences.py`.

## Prerequisites

- Trained models for target and avoid cell types (saved as `.h5` files)
- Models must be registered in the `MODEL_PATHS` dictionary in both scripts

## Step 1: Run Sequence Optimization

Run `mutate_sequences_target_avoid.py` to optimize sequences:

```bash
python mutate_sequences_target_avoid.py \
    --target L3.IT \
    --avoid L4.IT \
    [--input-fasta sequences.fasta] \
    [--n-seqs 200] \
    [--iterations 20] \
    [--output-prefix target_avoid_optimization]
```

### Required Arguments
- `--target`: Target cell type name (e.g., `L3.IT`)
- `--avoid`: Avoid cell type name (e.g., `L4.IT`)

### Optional Arguments
- `--input-fasta`: Path to FASTA file with initial sequences (default: generates random sequences)
- `--n-seqs`: Number of sequences to optimize (default: 200)
- `--iterations`: Number of saturation mutagenesis iterations (default: 20)
- `--output-prefix`: Prefix for output files (default: `target_avoid_optimization`)

### Output Files

The script creates two output files in the current directory:

1. **Results pickle file**: `{output_prefix}_{target}_vs_{avoid}_results.pkl`
   - Example: `target_avoid_optimization_L3.IT_vs_L4.IT_results.pkl`
   - Contains: `final_sequences`, `initial_sequences`, predictions, and optimization history

2. **Results plot**: `{output_prefix}_{target}_vs_{avoid}_results.pdf`
   - Example: `target_avoid_optimization_L3.IT_vs_L4.IT_results.pdf`
   - Contains: reward progression, target vs avoid scatter, reward distribution

### Model Files Used

Models are loaded from paths defined in `MODEL_PATHS` dictionary:

- **L3.IT vs L4.IT**:
  - Target model: `/home/arb1/cnn_pipeline/wandb/run-20251027_175203-oza0mj0j-Corces_Enh_L3.IT_fold1_first/files/model-best.h5`
  - Avoid model: `/home/arb1/cnn_pipeline/wandb/run-20251027_180454-0njg66rn-Corces_Enh_L4.IT_fold1_first/files/model-best.h5`

To add more cell type combinations, edit the `MODEL_PATHS` dictionary in `mutate_sequences_target_avoid.py`.

## Step 2: Analyze Final Sequences

Run `view_final_sequences.py` to extract and analyze the optimized sequences:

```bash
python view_final_sequences.py \
    --input-pkl target_avoid_optimization_L3.IT_vs_L4.IT_results.pkl \
    --target L3.IT \
    --avoid L4.IT \
    --output-fasta sequences.fasta \
    --output-csv predictions.csv \
    --output-plot frequency_plot.png \
    --output-prediction-plot prediction_plots.png \
    [--limit 100]
```

### Required Arguments
- `--input-pkl`: Path to the results pickle file from Step 1
- `--target`: Target cell type name (must match Step 1)
- `--avoid`: Avoid cell type name (must match Step 1)

### Optional Arguments
- `--output-fasta`: Export sequences to FASTA format
- `--output-csv`: Export predictions to CSV (includes: sequence, target_prediction, avoid_prediction, reward_prediction, gc_content)
- `--output-plot`: Generate frequency plot (per-position nucleotide frequencies)
- `--output-prediction-plot`: Generate prediction plots (target vs avoid scatter, reward histogram, GC vs reward scatter)
- `--limit`: If specified, select top N sequences by reward value (applies to FASTA and CSV only; plots show all sequences)

### Output Files

1. **FASTA file** (`sequences.fasta`): DNA sequences in FASTA format
2. **CSV file** (`predictions.csv`): Contains columns:
   - `sequence`: DNA sequence string
   - `target_prediction`: Target model prediction
   - `avoid_prediction`: Avoid model prediction
   - `reward_prediction`: Reward value (e^(target - avoid))
   - `gc_content`: GC content percentage
3. **Frequency plot** (`frequency_plot.png`): Per-position nucleotide frequency visualization
4. **Prediction plots** (`prediction_plots.png`): Three subplots:
   - Target vs Avoid predictions scatter plot
   - Reward distribution histogram
   - GC content vs Reward scatter plot

### Model Files Used

Same models as Step 1 (loaded from `MODEL_PATHS` dictionary in `view_final_sequences.py`).

## Example Complete Workflow

```bash
# Step 1: Optimize sequences
python mutate_sequences_target_avoid.py \
    --target L3.IT \
    --avoid L4.IT \
    --n-seqs 200 \
    --iterations 20 \
    --output-prefix target_avoid_optimization

# Step 2: Analyze and export results
python view_final_sequences.py \
    --input-pkl target_avoid_optimization_L3.IT_vs_L4.IT_results.pkl \
    --target L3.IT \
    --avoid L4.IT \
    --output-fasta optimized_sequences.fasta \
    --output-csv optimized_predictions.csv \
    --output-plot frequency_plot.png \
    --output-prediction-plot prediction_plots.png

# Optional: Get top 100 sequences
python view_final_sequences.py \
    --input-pkl target_avoid_optimization_L3.IT_vs_L4.IT_results.pkl \
    --target L3.IT \
    --avoid L4.IT \
    --limit 100 \
    --output-fasta top100_sequences.fasta \
    --output-csv top100_predictions.csv \
    --output-prediction-plot top100_plots.png
```

## Notes

- Both scripts use the same `MODEL_PATHS` dictionary - ensure cell type combinations are consistent
- The reward function is: `reward = e^(target_prediction - avoid_prediction)` (higher is better)
- When using `--limit`, CSV and FASTA contain only top N sequences, but plots show all sequences for context
- Models must be accessible at the paths specified in `MODEL_PATHS`
