"""gkm_config.py: Configuration management aligned with both CNN pipeline and gkm-SVM.

This module handles:
1. Loading CNN pipeline compatible YAML configs
2. Validating gkm-SVM specific parameters
3. Generating consistent run commands
4. Managing file paths between pipelines
"""

import os
import yaml
import logging
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_fasta_file(filepath: str) -> None:
    """Basic FASTA file validation."""
    if not os.path.exists(filepath):
        raise ValueError(f"File does not exist: {filepath}")
        
    with open(filepath) as f:
        first_line = f.readline().strip()
        if not first_line.startswith('>'):
            raise ValueError(f"Invalid FASTA format in {filepath}")


def calculate_class_weight(pos_count: int, neg_count: int, weighting_scheme: Optional[str] = None) -> float:
    """Calculate positive class weight based on sequence counts.
    
    Args:
        pos_count: Number of positive sequences
        neg_count: Number of negative sequences
        weighting_scheme: Strategy for weight calculation:
            - 'reciprocal': weight = (N/k) / n_i where N is total samples, k is classes (2)
            - 'proportional': weight = fraction of samples in other classes
            - None or 'none': no weighting (weight = 1.0)
        
    Returns:
        Weight value for positive class (SVM -w parameter)
    """
    if weighting_scheme in [None, 'none']:
        return 1.0
        
    total_count = pos_count + neg_count
    
    if weighting_scheme == 'reciprocal':
        # Chai's balancing method 
        balanced_count = total_count / 2
        return balanced_count / pos_count
        
    elif weighting_scheme == 'proportional':
        # Irene's balancing method
        return neg_count / total_count
        
    return 1.0  # Default to no weighting for unknown schemes

@dataclass
class GkmConfig:
    """Configuration for gkm-SVM models with CNN pipeline compatibility.
    
    Structure follows CNN pipeline while preserving gkm-SVM requirements:
    1. Project metadata (from CNN pipeline)
    2. Core gkm-SVM parameters
    3. Runtime settings
    4. Data paths (CNN pipeline format)
    """
    # CNN Pipeline Project Config
    project: str
    name: str = "gkm-svm"
    model_dir: Optional[str] = None
    output_prefix: Optional[str] = None
    dir: Optional[str] = None  # Directory for output files
    class_weight: Optional[str] = None  # Class weighting scheme ('none', 'reciprocal', or 'proportional')
    
    # Core gkm-SVM Parameters
    word_length: int = 11  # -l: word length (3-12)
    informed_cols: int = 7  # -k: informative columns (≤l) 
    max_mismatch: int = 3  # -d: max mismatches (≤4)
    kernel_type: int = 4  # -t: kernel function (0-5)
    
    # Kernel-Specific Parameters
    gamma: float = 1.0  # -g: RBF kernel gamma (t=3,5)
    init_decay: int = 50  # -M: wgkm initial decay (t=4,5)
    half_life: float = 50.0  # -H: wgkm half-life (t=4,5)
    
    # SVM Parameters
    regularization: float = 1.0  # -c: SVM regularization
    epsilon: float = 0.001  # -e: precision parameter
    pos_weight: float = 1.0  # -w: positive class weight
    cache_memory: float = 100.0  # -m: cache size (MB)
    use_shrinking: bool = False  # -s: use shrinking heuristics
    
    # Runtime Parameters
    verbosity: int = 2  # -v: verbosity (0-4)
    num_threads: int = 1  # -T: thread count (1,4,16)
    
    # CNN Pipeline Data Paths
    train_data_paths: List[Union[str, Dict[str, str]]] = field(default_factory=list)
    train_targets: List[Union[int, Dict[str, int]]] = field(default_factory=list)
    val_data_paths: List[Union[str, Dict[str, str]]] = field(default_factory=list)
    val_targets: List[Union[int, Dict[str, int]]] = field(default_factory=list)
    additional_val_data_paths: List[List[Union[str, Dict[str, str]]]] = field(default_factory=list)
    additional_val_targets: List[List[int]] = field(default_factory=list)
    
    # Add these fields to better support utils and trainer:
    genome_path: Optional[str] = None  # For BED conversion
    save_predictions: bool = True      # Control prediction file saving
    
    # Executable Paths
    gkm_executable: str = "gkmtrain"
    pred_executable: str = "gkmpredict"

    def __init__(self, word_length: int = 11, informed_cols: int = 7, **kwargs):
        if not (3 <= word_length <= 12) or informed_cols > word_length:
            raise ValueError(f"Invalid parameters: word_length must be 3-12 and informed_cols <= word_length")
            
        # Initialize core parameters
        self.word_length = word_length
        self.informed_cols = informed_cols
        self.max_mismatch = kwargs.get('max_mismatch', 3)
        self.kernel_type = kwargs.get('kernel_type', 4)
        
        # Initialize data paths
        self.train_data_paths = kwargs.get('train_data_paths', [])
        self.train_targets = kwargs.get('train_targets', [])
        self.val_data_paths = kwargs.get('val_data_paths', [])
        self.val_targets = kwargs.get('val_targets', [])
        self.additional_val_data_paths = kwargs.get('additional_val_data_paths', [])
        self.additional_val_targets = kwargs.get('additional_val_targets', [])
        
        # Initialize other parameters
        self.project = kwargs.get('project', 'default-project')
        self.name = kwargs.get('name', 'gkm-svm')
        self.model_dir = kwargs.get('model_dir')
        self.output_prefix = kwargs.get('output_prefix')
        self.dir = kwargs.get('dir')
        self.class_weight = kwargs.get('class_weight')

    def resolve_paths(self) -> None:
        """Resolve file paths to FASTA format."""
        self.train_data_paths = self._resolve_input_files(self.train_data_paths)
        if self.val_data_paths:
            self.val_data_paths = self._resolve_input_files(self.val_data_paths)
        if self.additional_val_data_paths:
            self.additional_val_data_paths = [
                self._resolve_input_files(paths) 
                for paths in self.additional_val_data_paths
            ]

    def _resolve_input_files(self, file_paths):
        """Convert input paths to FASTA files if needed."""
        resolved = []
        for entry in file_paths:
            # Handle dictionary paths
            if isinstance(entry, dict):
                if 'genome' in entry and 'intervals' in entry:
                    bed_file = entry['intervals']
                    fasta_path = f"{os.path.splitext(bed_file)[0]}.fa"
                    
                    # Check if valid FASTA exists
                    try:
                        validate_fasta_file(fasta_path)
                    except ValueError:
                        # Convert BED to FASTA
                        os.system(f"bedtools getfasta -name -fi {entry['genome']} -bed {bed_file} -fo {fasta_path}")
                    resolved.append(fasta_path)
                else:
                    # Direct path in dict
                    path = entry.get('path', entry)
                    validate_fasta_file(path)
                    resolved.append(path)
                    
            # Handle string paths
            else:
                if entry.endswith(('.bed', '.narrowPeak')):
                    fasta_path = f"{os.path.splitext(entry)[0]}.fa"
                    try:
                        validate_fasta_file(fasta_path)
                    except ValueError:
                        # Convert BED to FASTA
                        os.system(f"bedtools getfasta -name -fi {self.genome_path} -bed {entry} -fo {fasta_path}")
                    resolved.append(fasta_path)
                else:
                    # Should be direct FASTA path
                    validate_fasta_file(entry)
                    resolved.append(entry)
                    
        return resolved

    def validate(self) -> None:
        """Validate parameters against both gkm-SVM and CNN pipeline requirements."""
        # Core Parameter Validation
        self._validate_max_mismatch()
        self._validate_kernel_type()
        
        # Kernel-Specific Validation
        self._validate_kernel_params()
        
        # Runtime Validation
        self._validate_runtime_params()

        # Resolve paths and validate
        self.resolve_paths()

        # Data Path Validation
        self._validate_data_paths()

    def _validate_max_mismatch(self) -> None:
        """Validate maximum mismatches (d) parameter."""
        if not 0 <= self.max_mismatch <= 4:
            raise ValueError(f"max_mismatch (d={self.max_mismatch}) must be between 0 and 4")
            
        if self.max_mismatch > self.informed_cols:
            raise ValueError(
                f"max_mismatch (d={self.max_mismatch}) must be <= "
                f"informed_cols (k={self.informed_cols})"
            )

    def _validate_kernel_type(self) -> None:
        """Validate kernel type (t) parameter."""
        if not 0 <= self.kernel_type <= 5:
            raise ValueError(f"kernel_type (t={self.kernel_type}) must be between 0 and 5")

    def _validate_kernel_params(self) -> None:
        """Validate kernel-specific parameters."""
        # RBF Kernel Parameters (t=3,5)
        if self.kernel_type in [3, 5]:
            if self.gamma <= 0:
                raise ValueError(f"gamma (g={self.gamma}) must be positive for RBF kernels")
                
        # Weighted Kernel Parameters (t=4,5)
        if self.kernel_type in [4, 5]:
            if not 0 <= self.init_decay <= 255:
                raise ValueError(f"init_decay (M={self.init_decay}) must be between 0 and 255")
            if self.half_life <= 0:
                raise ValueError(f"half_life (H={self.half_life}) must be positive")

    def _validate_runtime_params(self) -> None:
        """Validate runtime parameters."""
        if not 0 <= self.verbosity <= 4:
            raise ValueError(f"verbosity (v={self.verbosity}) must be between 0 and 4")
            
        if self.num_threads not in [1, 4, 16]:
            raise ValueError(f"num_threads (T={self.num_threads}) must be 1, 4 or 16")

        if self.regularization <= 0:
            raise ValueError(f"regularization (c={self.regularization}) must be positive")
            
        if self.epsilon <= 0:
            raise ValueError(f"epsilon (e={self.epsilon}) must be positive")
            
        if self.pos_weight <= 0:
            raise ValueError(f"pos_weight (w={self.pos_weight}) must be positive")
            
        if self.cache_memory <= 0:
            raise ValueError(f"cache_memory (m={self.cache_memory}) must be positive")

    def _validate_data_paths(self) -> None:
        """Validate CNN pipeline data paths."""
        # Check path counts match
        if len(self.train_data_paths) != len(self.train_targets):
            raise ValueError("Number of training paths must match number of targets")
            
        if len(self.val_data_paths) != len(self.val_targets):
            raise ValueError("Number of validation paths must match number of targets")
            
        if len(self.additional_val_data_paths) != len(self.additional_val_targets):
            raise ValueError("Number of additional validation sets must match number of targets")

        # Validate file existence
        for paths in [self.train_data_paths, self.val_data_paths]:
            for path in paths:
                file_path = self._get_file_path(path)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file not found: {file_path}")

        for path_set in self.additional_val_data_paths:
            for path in path_set:
                file_path = self._get_file_path(path)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Additional validation file not found: {file_path}")

    def _get_file_path(self, path_spec: Union[str, Dict[str, str]]) -> str:
        """Extract file path from CNN pipeline path specification."""
        if isinstance(path_spec, dict):
            if 'intervals' in path_spec:
                return path_spec['intervals']
            raise ValueError(f"Invalid path specification: {path_spec}")
        return path_spec


    def get_run_prefix(self) -> str:
        """Generate unique prefix incorporating key parameters and config name."""
        if self.output_prefix:
            return self.output_prefix
            
        # Core parameters
        params = [
            f"l{self.word_length}",
            f"k{self.informed_cols}",
            f"d{self.max_mismatch}",
            f"t{self.kernel_type}",
            f"c{self.regularization}"
        ]
        
        # Kernel-specific parameters
        if self.kernel_type in [3, 5]:  # RBF
            params.append(f"g{self.gamma}")
            
        if self.kernel_type in [4, 5]:  # Weighted
            params.extend([
                f"M{self.init_decay}",
                f"H{self.half_life}"
            ])
        
        # Add config name as suffix
        prefix = f"gkm-{'_'.join(params)}-{self.name}"
        if self.model_dir:
            return os.path.join(self.model_dir, prefix)
        return prefix

    def _validate_resolved_paths(self) -> None:
        """Validate all resolved paths exist and are in correct format."""
        for path in self.train_data_paths + self.val_data_paths:
            validate_fasta_file(path)
            
        for path_set in self.additional_val_data_paths:
            for path in path_set:
                validate_fasta_file(path)
    
    def get_train_files(self) -> Tuple[List[str], List[str]]:
        """Get positive and negative training files.
        
        Returns:
            Tuple of (positive_files, negative_files)
        """
        pos_files = [p for p, t in zip(self.train_data_paths, self.train_targets) if t == 1]
        neg_files = [p for p, t in zip(self.train_data_paths, self.train_targets) if t == 0]
        
        if not pos_files:
            raise ValueError("No positive training files found")
        if not neg_files:
            raise ValueError("No negative training files found")
            
        return pos_files, neg_files


    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GkmConfig':
        """Create config from CNN pipeline YAML format."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path) as f:
            yaml_dict = yaml.safe_load(f)

        # Extract CNN pipeline fields
        config_dict = {
            'project': yaml_dict.get('project', {}).get('value', 'default-project'),
            'name': yaml_dict.get('name', {}).get('value', 'gkm-svm'),
            'model_dir': yaml_dict.get('model_dir', {}).get('value'),
            'output_prefix': yaml_dict.get('output_prefix', {}).get('value'),
            'dir': yaml_dict.get('dir', {}).get('value'),
            'class_weight': yaml_dict.get('class_weight', {}).get('value', 'none'),  # Add class_weight
                       
            # Data paths
            'train_data_paths': yaml_dict.get('train_data_paths', {}).get('value', []),
            'train_targets': yaml_dict.get('train_targets', {}).get('value', []),
            'val_data_paths': yaml_dict.get('val_data_paths', {}).get('value', []),
            'val_targets': yaml_dict.get('val_targets', {}).get('value', []),
            'additional_val_data_paths': yaml_dict.get('additional_val_data_paths', {}).get('value', []),
            'additional_val_targets': yaml_dict.get('additional_val_targets', {}).get('value', [])
        }

        # Create and validate config
        config = cls(**config_dict)
        
        # Validate class_weight value if specified
        if config.class_weight not in [None, 'none', 'reciprocal', 'proportional']:
            raise ValueError(f"Invalid class_weight value: {config.class_weight}. "
                           "Must be one of: none, reciprocal, proportional")
        
        config.validate()
        return config

    def get_train_cmd(self, pos_file: str, neg_file: str, out_prefix: str) -> List[str]:
        """Build gkmtrain command with exact parameter matching."""
        cmd = [
            self.gkm_executable,
            # Core parameters
            '-l', str(self.word_length),
            '-k', str(self.informed_cols),
            '-d', str(self.max_mismatch),
            '-t', str(self.kernel_type)
        ]
        
        # Kernel-specific parameters
        if self.kernel_type in [3, 5]:  # RBF
            cmd.extend(['-g', str(self.gamma)])
            
        if self.kernel_type in [4, 5]:  # Weighted
            cmd.extend([
                '-M', str(self.init_decay),
                '-H', str(self.half_life)
            ])
            
        # General parameters
        cmd.extend([
            '-c', str(self.regularization),
            '-e', str(self.epsilon),
            '-w', str(self.pos_weight),
            '-m', str(self.cache_memory),
            '-v', str(self.verbosity),
            '-T', str(self.num_threads)
        ])
        
        # Boolean flags
        if self.use_shrinking:
            cmd.append('-s')
            
        # Input/output paths
        cmd.extend([pos_file, neg_file, out_prefix])
        
        return cmd

    def get_predict_cmd(self, test_file: str, model_file: str, out_file: str) -> List[str]:
        """Build gkmpredict command with exact parameter matching."""
        return [
            self.pred_executable,
            test_file,
            model_file,
            out_file,
            '-v', str(self.verbosity),
            '-T', str(self.num_threads)
        ]
