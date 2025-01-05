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
from typing import List, Dict, Optional, Union, Any, Tuple, Set
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Type alias for path-like objects
PathLike = Union[str, Path]

class GkmUtilsError(Exception):
    """Base exception class for gkm-utils errors."""
    pass

class FilePatterns:
    """Constants for file naming patterns."""
    MODEL_FILE = "{prefix}.model.txt"
    PREDICTION_FILE = "{model_name}_{prefix}_{test_name}_predictions.txt"
    RESULTS_FILE = "{model_prefix}_results.json"
    COMBINED_FASTA = "{name}-{class_label}.fa"

@dataclass
class GkmParams:
    """Container for gkm-SVM parameters with validation."""
    word_length: int = field(default=11, metadata={'min': 3, 'max': 12})
    informed_cols: int = field(default=7, metadata={'max_ref': 'word_length'})
    max_mismatch: int = field(default=3, metadata={'min': 0, 'max': 4})
    kernel_type: int = field(default=4, metadata={'allowed': [0, 1, 2, 3, 4, 5]})
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_word_length()
        self._validate_informed_cols()
        self._validate_max_mismatch()
        self._validate_kernel_type()
        
    def _validate_word_length(self) -> None:
        if not self.metadata['min'] <= self.word_length <= self.metadata['max']:
            raise ValueError(f"word_length must be between {self.metadata['min']} and {self.metadata['max']}")
            
    def _validate_informed_cols(self) -> None:
        if self.informed_cols > self.word_length:
            raise ValueError(f"informed_cols ({self.informed_cols}) must be <= word_length ({self.word_length})")
            
    def _validate_max_mismatch(self) -> None:
        if not self.metadata['min'] <= self.max_mismatch <= self.metadata['max']:
            raise ValueError(f"max_mismatch must be between {self.metadata['min']} and {self.metadata['max']}")
        if self.max_mismatch > self.informed_cols:
            raise ValueError(f"max_mismatch ({self.max_mismatch}) must be <= informed_cols ({self.informed_cols})")
            
    def _validate_kernel_type(self) -> None:
        if self.kernel_type not in self.metadata['allowed']:
            raise ValueError(f"kernel_type must be one of {self.metadata['allowed']}")

class FASTAConverter:
    """Handles conversion between file formats."""
    
    @staticmethod
    def bed_to_fasta(bed_file: PathLike, 
                     genome_file: PathLike,
                     output_path: PathLike) -> None:
        """Convert BED file to FASTA using bedtools."""
        bed_file = Path(bed_file)
        genome_file = Path(genome_file)
        output_path = Path(output_path)
        
        # Validate input and output paths
        Validator.validate_input_file(bed_file)
        Validator.validate_input_file(genome_file)
        Validator.validate_output_path(output_path)

        # Check if bedtools is available
        if subprocess.run(['which', 'bedtools'], capture_output=True).returncode != 0:
            raise GkmUtilsError("bedtools command not found. Please install bedtools")

        try:
            # Check if bed file has a name column
            with open(bed_file) as f:
                has_name_column = len(f.readline().strip().split('\t')) >= 4
            
            # Construct bedtools command
            cmd = ['bedtools', 'getfasta', 
                  '-fi', str(genome_file), 
                  '-bed', str(bed_file)]
            if has_name_column:
                cmd.append('-name')
            cmd.extend(['-fo', str(output_path)])
                
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise GkmUtilsError(f"bedtools command failed: {result.stderr}")
            
            # Validate the output
            if not output_path.exists():
                raise GkmUtilsError(f"FASTA file not created: {output_path}")
            FASTAReader.validate_format(output_path)
            
        except Exception as e:
            raise GkmUtilsError(f"BED to FASTA conversion failed: {str(e)}")

class Validator:
    """Validates file paths and sequences for gkm-SVM processing."""
    
    @staticmethod
    def validate_training_files(pos_files: List[PathLike], neg_files: List[PathLike]) -> None:
        """Comprehensive validation of training file sets."""
        for files in [pos_files, neg_files]:
            for file in files:
                Validator.validate_training_file(file)
    
    @staticmethod
    def validate_training_file(filepath: PathLike) -> None:
        """Comprehensive validation for a single training file."""
        filepath = Path(filepath)
        Validator.validate_input_file(filepath)
        FASTAReader.validate_format(filepath)
           
    @staticmethod
    def validate_input_file(filepath: PathLike) -> None:
        """Check if input file exists and is readable."""
        path = Path(filepath)
        if not path.exists():
            raise GkmUtilsError(f"File does not exist: {filepath}")
        if not os.access(path, os.R_OK):
            raise GkmUtilsError(f"File is not readable: {filepath}")
            
    @staticmethod
    def validate_output_path(filepath: PathLike) -> None:
        """Validate output path is writable."""
        path = Path(filepath)
        if path.exists() and not os.access(path, os.W_OK):
            raise GkmUtilsError(f"Path exists but is not writable: {filepath}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise GkmUtilsError(f"Cannot create parent directories for {filepath}: {e}")


@dataclass
class GkmConfig:
    """Configuration for gkm-SVM models with CNN pipeline compatibility."""
    # Project Config
    project: str
    name: str = "gkm-svm"
    model_dir: Optional[str] = None
    output_prefix: Optional[str] = None
    dir: Optional[str] = None
    class_weight: Optional[str] = None
    
    # Core Parameters 
    params: GkmParams = field(default_factory=GkmParams)
    
    # Runtime Parameters
    verbosity: int = 2
    num_threads: int = 1
    
    # Data Paths
    train_data_paths: List[Union[str, Dict[str, str]]] = field(default_factory=list)
    train_targets: List[Union[int, Dict[str, int]]] = field(default_factory=list)
    val_data_paths: List[Union[str, Dict[str, str]]] = field(default_factory=list)
    val_targets: List[Union[int, Dict[str, int]]] = field(default_factory=list)
    additional_val_data_paths: List[List[Union[str, Dict[str, str]]]] = field(default_factory=list)
    additional_val_targets: List[List[int]] = field(default_factory=list)
    
    # Additional Fields
    genome_path: Optional[str] = None
    save_predictions: bool = True
    
    # Executable Paths
    gkm_executable: str = "gkmtrain"
    pred_executable: str = "gkmpredict"

    def get_model_dir(self) -> Path:
        """Get path to model directory."""
        base_dir = Path(self.dir) if self.dir else Path.cwd()
        model_dir = base_dir / "lsgkm" / self.name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
        
    def get_prediction_dir(self) -> Path:
        """Get path to predictions directory."""
        return self.get_model_dir() / "predictions"
    
    def get_prediction_path(self, test_file: PathLike, prefix: Optional[str] = None) -> Path:
        """Generate path for prediction output file."""
        pred_dir = self.get_prediction_dir()
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        test_name = Path(test_file).stem
        if prefix:
            return pred_dir / FilePatterns.PREDICTION_FILE.format(
                model_name=self.name,
                prefix=prefix,
                test_name=test_name
            )
        return pred_dir / f"{self.name}_{test_name}_predictions.txt"

    def resolve_file_path(self, path_spec: Union[str, Dict[str, str]]) -> Path:
        """Resolve file path from CNN pipeline format."""
        if isinstance(path_spec, dict):
            if 'intervals' in path_spec:
                return Path(path_spec['intervals'])
            raise ValueError(f"Invalid path specification: {path_spec}")
        return Path(path_spec)
    
    def resolve_paths(self) -> None:
        """Resolve and validate all input paths."""
        self.train_data_paths = [self.resolve_file_path(p) for p in self.train_data_paths]
        self.val_data_paths = [self.resolve_file_path(p) for p in self.val_data_paths]
        
        if self.additional_val_data_paths:
            self.additional_val_data_paths = [
                [self.resolve_file_path(p) for p in paths] 
                for paths in self.additional_val_data_paths
            ]
        
        # Validate all resolved paths
        self._validate_resolved_paths()

    def _validate_resolved_paths(self) -> None:
        """Validate all resolved paths exist and are in correct format."""
        for path in self.train_data_paths:
            Validator.validate_training_file(path)
            
        for path in self.val_data_paths:
            Validator.validate_training_file(path)
            
        for path_set in self.additional_val_data_paths:
            for path in path_set:
                Validator.validate_training_file(path)

    def get_train_files(self) -> Tuple[List[Path], List[Path]]:
        """Get positive and negative training files."""
        pos_files = [p for p, t in zip(self.train_data_paths, self.train_targets) if t == 1]
        neg_files = [p for p, t in zip(self.train_data_paths, self.train_targets) if t == 0]
        
        if not pos_files:
            raise ValueError("No positive training files found")
        if not neg_files:
            raise ValueError("No negative training files found")
            
        return pos_files, neg_files

    def get_val_files(self) -> Tuple[List[Path], List[Path]]:
        """Get positive and negative validation files."""
        pos_files = [p for p, t in zip(self.val_data_paths, self.val_targets) if t == 1]
        neg_files = [p for p, t in zip(self.val_data_paths, self.val_targets) if t == 0]
        return pos_files, neg_files

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

        # Resolve additional validation sets if present
        if self.additional_val_data_paths:
            resolved_additional = []
            for paths in self.additional_val_data_paths:
                resolved = resolve_input_files(paths, self.genome_path)
                resolved_additional.append(resolved)
            self.additional_val_data_paths = resolved_additional
            
        # Validate all resolved paths
        self._validate_resolved_paths()
    
    def get_prediction_path(self, test_file: str, prefix: str = None) -> str:
        """Standardize prediction file path generation."""
        model_base_dir = Path(self.dir) if self.dir else Path.cwd()
        pred_dir = model_base_dir / "lsgkm" / self.name / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        test_name = Path(test_file).stem
        if prefix:
            return str(pred_dir / f"{self.name}_{prefix}_{test_name}_predictions.txt")
        return str(pred_dir / f"{self.name}_{test_name}_predictions.txt")

    def validate(self) -> None:
        """Validate parameters against both gkm-SVM and CNN pipeline requirements."""
        # Core Parameter Validation
        self._validate_word_length()
        self._validate_informed_cols()
        self._validate_max_mismatch()
        self._validate_kernel_type()
        
        # Kernel-Specific Validation
        self._validate_kernel_params()
        
        # Runtime Validation
        self._validate_runtime_params()
        
        # Data Path Validation
        self._validate_data_paths()

 
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

    def _validate_resolved_paths(self) -> None:
        """Validate all resolved paths exist and are in correct format."""
        for path in self.train_data_paths:
            Validator.validate_file(path)
            
        for path in self.val_data_paths:
            Validator.validate_file(path)
            
        for path_set in self.additional_val_data_paths:
            for path in path_set:
                Validator.validate_file(path)

    def _get_file_path(self, path_spec: Union[str, Dict[str, str]]) -> str:
        """Extract file path from CNN pipeline path specification."""
        if isinstance(path_spec, dict):
            if 'intervals' in path_spec:
                return path_spec['intervals']
            raise ValueError(f"Invalid path specification: {path_spec}")
        return path_spec
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GkmConfig':
        """Create config from CNN pipeline YAML format.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Initialized GkmConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing or invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path) as f:
            yaml_dict = yaml.safe_load(f)

        def get_value(key: str, default: Any = None) -> Any:
            """Helper to extract values from nested YAML structure."""
            entry = yaml_dict.get(key, {})
            if isinstance(entry, dict):
                return entry.get('value', default)
            return entry

        # Extract gkm-SVM specific parameters
        params = GkmParams(
            word_length=get_value('word_length', 11),
            informed_cols=get_value('informed_cols', 7),
            max_mismatch=get_value('max_mismatch', 3),
            kernel_type=get_value('kernel_type', 4)
        )

        # Extract CNN pipeline paths and targets
        train_data = get_value('train_data_paths', [])
        train_targets = get_value('train_targets', [])
        val_data = get_value('val_data_paths', [])
        val_targets = get_value('val_targets', [])
        additional_val_data = get_value('additional_val_data_paths', [])
        additional_val_targets = get_value('additional_val_targets', [])

        # Validate path/target pairs
        if len(train_data) != len(train_targets):
            raise ValueError("Number of training paths must match number of targets")
        if len(val_data) != len(val_targets):
            raise ValueError("Number of validation paths must match number of targets")
        if len(additional_val_data) != len(additional_val_targets):
            raise ValueError("Number of additional validation sets must match targets")

        # Extract runtime parameters
        class_weight = get_value('class_weight', 'none')
        if class_weight not in [None, 'none', 'reciprocal', 'proportional']:
            raise ValueError(f"Invalid class_weight: {class_weight}. "
                           "Must be one of: none, reciprocal, proportional")

        # Create config with all parameters
        config = cls(
            # Project metadata
            project=get_value('project', 'default-project'),
            name=get_value('name', 'gkm-svm'),
            model_dir=get_value('model_dir'),
            output_prefix=get_value('output_prefix'),
            dir=get_value('dir'),
            class_weight=class_weight,
            
            # Core parameters and data paths
            params=params,
            train_data_paths=train_data,
            train_targets=train_targets,
            val_data_paths=val_data,
            val_targets=val_targets,
            additional_val_data_paths=additional_val_data,
            additional_val_targets=additional_val_targets,
            
            # Runtime parameters
            verbosity=get_value('verbosity', 2),
            num_threads=get_value('num_threads', 1),
            
            # Optional parameters
            genome_path=get_value('genome_path'),
            save_predictions=get_value('save_predictions', True)
        )

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

class FASTAReader:
    """Handles reading and validation of FASTA files."""
    
    @staticmethod
    def count_sequences(fasta_file: PathLike) -> int:
        """Count number of sequences in a FASTA file."""
        count = 0
        with open(fasta_file) as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count
    
    @staticmethod
    def validate_format(fasta_file: PathLike) -> None:
        """Validate FASTA file format."""
        with open(fasta_file) as f:
            first_line = f.readline().strip()
            if not first_line.startswith('>'):
                raise GkmUtilsError(f"Invalid FASTA format in {fasta_file}")

class FASTAConverter:
    """Handles conversion between file formats."""
    
    @staticmethod
    def bed_to_fasta(bed_file: PathLike, 
                     genome_file: PathLike,
                     output_path: PathLike) -> None:
        """Convert BED file to FASTA using bedtools."""
        bed_file = Path(bed_file)
        genome_file = Path(genome_file)
        output_path = Path(output_path)
        
        # Validate paths
        Validator.validate_input_file(bed_file)
        Validator.validate_input_file(genome_file)
        Validator.validate_output_path(output_path)

        # Check bedtools
        if subprocess.run(['which', 'bedtools'], capture_output=True).returncode != 0:
            raise GkmUtilsError("bedtools command not found")

        try:
            # Check for name column
            with open(bed_file) as f:
                has_name_column = len(f.readline().strip().split('\t')) >= 4
            
            cmd = ['bedtools', 'getfasta', 
                  '-fi', str(genome_file), 
                  '-bed', str(bed_file)]
            if has_name_column:
                cmd.append('-name')
            cmd.extend(['-fo', str(output_path)])
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise GkmUtilsError(f"bedtools failed: {result.stderr}")
            
            if not output_path.exists():
                raise GkmUtilsError(f"FASTA file not created: {output_path}")
            FASTAReader.validate_format(output_path)
            
        except Exception as e:
            raise GkmUtilsError(f"BED to FASTA conversion failed: {str(e)}")


