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

class GkmError(Exception):
    """Base exception class for gkm-utils errors."""
    pass

class PathResolutionError(Exception):
    """Custom exception for path resolution errors."""
    pass

@dataclass
class DataPath:
    """Container for input data paths."""
    file_path: Path
    genome_path: Optional[Path] = None
    
    @property
    def is_bed(self) -> bool:
        return bool(self.genome_path)

@dataclass
class ValidationResult:
    """Container for validation results with errors and warnings."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add error message and set is_valid to False."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add warning message."""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another ValidationResult into this one."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        
class PathManager:
    """Centralized path handling for gkm-SVM configuration."""
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize PathManager with optional base directory.
        
        Args:
            base_dir: Base directory for relative paths. Uses CWD if not specified.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
    
    def resolve_data_path(self, path_spec: Union[str, Dict[str, str]]) -> DataPath:
        """Resolve CNN pipeline path format to DataPath object.
        
        Args:
            path_spec: Either a string path to FASTA file or dict with genome/intervals paths
            
        Returns:
            DataPath object with resolved paths
            
        Raises:
            PathResolutionError: If path format is invalid or files don't exist
        """
        try:
            if isinstance(path_spec, str):
                # Direct FASTA file path
                return DataPath(file_path=self._resolve_path(path_spec))
            
            elif isinstance(path_spec, dict):
                # BED file with genome
                if 'intervals' not in path_spec:
                    raise PathResolutionError(f"Missing 'intervals' in path specification: {path_spec}")
                    
                file_path = self._resolve_path(path_spec['intervals'])
                genome_path = self._resolve_path(path_spec['genome']) if 'genome' in path_spec else None
                
                return DataPath(file_path=file_path, genome_path=genome_path)
            
            else:
                raise PathResolutionError(f"Invalid path specification type: {type(path_spec)}")
                
        except Exception as e:
            raise PathResolutionError(f"Failed to resolve path {path_spec}: {str(e)}")
    
    def resolve_model_dir(self, model_name: str) -> Path:
        """Get path to model directory, creating it if needed.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model directory
        """
        model_dir = self.base_dir / "lsgkm" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def resolve_prediction_dir(self, model_name: str) -> Path:
        """Get path to predictions directory, creating it if needed.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to predictions directory
        """
        pred_dir = self.resolve_model_dir(model_name) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        return pred_dir
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Convert path string to Path object, resolving relative to base_dir.
        
        Args:
            path: Path string or Path object
            
        Returns:
            Resolved Path object
            
        Raises:
            PathResolutionError: If path doesn't exist
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            resolved_path = path_obj
        else:
            resolved_path = self.base_dir / path_obj
            
        if not resolved_path.exists():
            raise PathResolutionError(f"Path does not exist: {resolved_path}")
            
        return resolved_path

    def get_prediction_path(self, model_name: str, test_file: Union[str, Path], prefix: Optional[str] = None) -> Path:
        """Generate standardized path for prediction output file.
        
        Args:
            model_name: Name of the model
            test_file: Path to test file being predicted
            prefix: Optional prefix for prediction file name
            
        Returns:
            Path where predictions should be saved
        """
        pred_dir = self.resolve_prediction_dir(model_name)
        test_name = Path(test_file).stem
        
        if prefix:
            return pred_dir / f"{model_name}_{prefix}_{test_name}_predictions.txt"
        return pred_dir / f"{model_name}_{test_name}_predictions.txt"

    def get_model_path(self, model_name: str, run_prefix: str) -> Path:
        """Generate standardized path for model file.
        
        Args:
            model_name: Name of the model
            run_prefix: Prefix incorporating run parameters
            
        Returns:
            Path where model should be saved
        """
        model_dir = self.resolve_model_dir(model_name)
        return model_dir / f"{run_prefix}.model.txt"

    def get_combined_fasta_paths(self, model_name: str, class_label: str) -> Path:
        """Generate paths for combined FASTA files.
        
        Args:
            model_name: Name of the model
            class_label: Class label (e.g. "pos" or "neg")
            
        Returns:
            Path for combined FASTA file
        """
        model_dir = self.resolve_model_dir(model_name)
        return model_dir / f"{model_name}-{class_label}.fa"

        
class Validator:
    """Consolidated validation utilities."""
    
    @staticmethod
    def validate_training_files(pos_files: List[PathLike], neg_files: List[PathLike]) -> None:
        """Validate all training files."""
        for files in [pos_files, neg_files]:
            for file in files:
                Validator.validate_training_file(file)
    
    @staticmethod 
    def validate_training_file(filepath: PathLike) -> None:
        """Validate a single training file."""
        filepath = Path(filepath)
        Validator.validate_input_file(filepath)
        FileHandler.validate_fasta(filepath)
           
    @staticmethod
    def validate_input_file(filepath: PathLike) -> None:
        """Check if input file exists and is readable."""
        path = Path(filepath)
        if not path.exists():
            raise GkmError(f"File does not exist: {filepath}")
        if not os.access(path, os.R_OK):
            raise GkmError(f"File is not readable: {filepath}")
            
    @staticmethod
    def validate_output_path(filepath: PathLike) -> None:
        """Validate output path is writable."""
        path = Path(filepath)
        if path.exists() and not os.access(path, os.W_OK):
            raise GkmError(f"Path exists but is not writable: {filepath}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise GkmError(f"Cannot create parent directories for {filepath}: {e}")


class FASTAHandler:
    """Handles reading and validation of FASTA files."""
    
    @staticmethod
    def combine_files(input_files: List[PathLike], output_file: PathLike) -> int:
        """Combine multiple FASTA files into one and return sequence count.
        
        Args:
            input_files: List of input FASTA files
            output_file: Path for combined output
            
        Returns:
            Number of sequences in combined file
        """
        count = 0
        with open(output_file, 'w') as outfile:
            for file in input_files:
                with open(file) as infile:
                    for line in infile:
                        if line.startswith('>'):
                            count += 1
                        outfile.write(line)
        return count
        
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
                raise GkmError(f"Invalid FASTA format in {fasta_file}")

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
            raise GkmError("bedtools command not found")

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
                raise GkmError(f"bedtools failed: {result.stderr}")
            
            if not output_path.exists():
                raise GkmError(f"FASTA file not created: {output_path}")
            FASTAHandler.validate_format(output_path)
            
        except Exception as e:
            raise GkmError(f"BED to FASTA conversion failed: {str(e)}")


@dataclass
class GkmParams:
    """Container for gkm-SVM specific parameters."""
    word_length: int = 11          # -l: word length (default 11)
    informed_cols: int = 7         # -k: number of informative columns (default 7) 
    max_mismatch: int = 3         # -d: maximum number of mismatches (default 3)
    kernel_type: int = 4          # -t: kernel type, one of [0,1,2,3,4,5] (default 4 wgkm)
    gamma: float = 1.0           # -g: gamma parameter for RBF kernel (t=3,5)
    init_decay: int = 50         # -M: initial value for weight decay (t=4,5)
    half_life: float = 50.0      # -H: half-life parameter for weight decay (t=4,5)
    regularization: float = 1.0   # -c: regularization parameter (default 1.0)
    epsilon: float = 0.001       # -e: precision parameter (default 0.001)
    pos_weight: float = 1.0      # -w: weight for positive class (default 1.0)
    cache_memory: float = 100.0  # -m: cache memory size in MB (default 100.0)
    use_shrinking: bool = False  # -s: use shrinking heuristics if True
    
    # Path to executable
    gkm_executable: str = "gkmtrain"  # Path to gkmtrain executable
    pred_executable: str = "gkmpredict"  # Path to gkmpredict executable

    
@dataclass
class GkmConfig:
    """Configuration for gkm-SVM models with CNN pipeline compatibility."""
    # Project Config
    project: str
    name: str = "gkm-svm"
    dir: Optional[str] = None
    output_prefix: Optional[str] = None
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
    
    def __post_init__(self):
        """Initialize path manager and resolve data paths."""
        self.path_manager = PathManager(self.dir)
        
        # Convert all data paths to DataPath objects
        self._train_paths = [
            self.path_manager.resolve_data_path(path)
            for path in self.train_data_paths
        ]
        self._val_paths = [
            self.path_manager.resolve_data_path(path)
            for path in self.val_data_paths
        ]
        self._additional_val_paths = [
            [self.path_manager.resolve_data_path(path) for path in paths]
            for paths in self.additional_val_data_paths
        ]

    def get_model_dir(self) -> Path:
        """Get path to model directory."""
        base_dir = Path(self.dir) if self.dir else Path.cwd()
        model_dir = base_dir / "lsgkm" / self.name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
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
    
    def get_prediction_path(self, test_file: str, prefix: str = None) -> str:
        """Standardize prediction file path generation."""
        model_base_dir = Path(self.dir) if self.dir else Path.cwd()
        pred_dir = model_base_dir / "lsgkm" / self.name / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        test_name = Path(test_file).stem
        if prefix:
            return str(pred_dir / f"{self.name}_{prefix}_{test_name}_predictions.txt")
        return str(pred_dir / f"{self.name}_{test_name}_predictions.txt")

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
        """
        with open(yaml_path) as f:
            yaml_dict = yaml.safe_load(f)

        def get_value(key: str, default: Any = None) -> Any:
            """Helper to extract values from nested YAML structure."""
            entry = yaml_dict.get(key, {})
            if isinstance(entry, dict):
                return entry.get('value', default)
            return entry

        # Extract core parameters
        params = GkmParams(
            word_length=get_value('word_length', 11),
            informed_cols=get_value('informed_cols', 7),
            max_mismatch=get_value('max_mismatch', 3),
            kernel_type=get_value('kernel_type', 4)
        )

        # Create config instance
        return cls(
            project=get_value('project', 'default-project'),
            name=get_value('name', 'gkm-svm'),
            dir=get_value('dir'),
            output_prefix=get_value('output_prefix'),
            class_weight=get_value('class_weight'),
            params=params,
            train_data_paths=get_value('train_data_paths', []),
            train_targets=get_value('train_targets', []),
            val_data_paths=get_value('val_data_paths', []),
            val_targets=get_value('val_targets', []),
            additional_val_data_paths=get_value('additional_val_data_paths', []),
            additional_val_targets=get_value('additional_val_targets', []),
            verbosity=get_value('verbosity', 2),
            num_threads=get_value('num_threads', 1)
        )

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


    def validate(self) -> None:
        """Validate entire configuration.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        # Run all validations and collect results
        result = ValidationResult(is_valid=True)
        
        # Core parameter validation
        param_result = self._validate_parameters()
        result.merge(param_result)
        
        # Data configuration validation
        data_result = self._validate_data_config()
        result.merge(data_result)
        
        # Path validation
        path_result = self._validate_paths()
        result.merge(path_result)
        
        # Handle validation outcome
        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)
                
        if not result.is_valid:
            raise ConfigValidationError("\n".join(result.errors))

    def _validate_parameters(self) -> ValidationResult:
        """Validate GkmParams configuration."""
        result = ValidationResult(is_valid=True)
        
        # Validate word length
        if not 3 <= self.params.word_length <= 12:
            result.add_error(
                f"word_length must be between 3 and 12, got {self.params.word_length}"
            )
            
        # Validate informed columns
        if self.params.informed_cols > self.params.word_length:
            result.add_error(
                f"informed_cols ({self.params.informed_cols}) must be <= "
                f"word_length ({self.params.word_length})"
            )
            
        # Validate max mismatch
        if not 0 <= self.params.max_mismatch <= 4:
            result.add_error(
                f"max_mismatch must be between 0 and 4, got {self.params.max_mismatch}"
            )
        if self.params.max_mismatch > self.params.informed_cols:
            result.add_error(
                f"max_mismatch ({self.params.max_mismatch}) must be <= "
                f"informed_cols ({self.params.informed_cols})"
            )
            
        # Validate kernel type
        if self.params.kernel_type not in [0, 1, 2, 3, 4, 5]:
            result.add_error(
                f"kernel_type must be one of [0,1,2,3,4,5], got {self.params.kernel_type}"
            )
            
        return result

    def _validate_data_config(self) -> ValidationResult:
        """Validate data configuration."""
        result = ValidationResult(is_valid=True)
        
        # Validate path/target pairs
        if len(self.train_data_paths) != len(self.train_targets):
            result.add_error("Number of training paths must match number of targets")
            
        if len(self.val_data_paths) != len(self.val_targets):
            result.add_error("Number of validation paths must match number of targets")
            
        if len(self.additional_val_data_paths) != len(self.additional_val_targets):
            result.add_error("Number of additional validation sets must match number of targets")
            
        # Validate class weights
        if self.class_weight not in [None, 'none', 'reciprocal', 'proportional']:
            result.add_error(
                f"Invalid class_weight: {self.class_weight}. "
                "Must be one of: none, reciprocal, proportional"
            )
            
        return result

    def _validate_paths(self) -> ValidationResult:
        """Validate all data paths."""
        result = ValidationResult(is_valid=True)
        
        # Validate training paths
        for path in self._train_paths:
            if not path.file_path.exists():
                result.add_error(f"Training file does not exist: {path.file_path}")
            # Validate FASTA format only if file exists
            elif not path.genome_path:  # Only validate FASTA files, not BED files
                try:
                    FASTAHandler.validate_format(path.file_path)
                except GkmError as e:
                    result.add_error(str(e))
                    
        # Validate validation paths
        for path in self._val_paths:
            if not path.file_path.exists():
                result.add_error(f"Validation file does not exist: {path.file_path}")
            elif not path.genome_path:
                try:
                    FASTAHandler.validate_format(path.file_path)
                except GkmError as e:
                    result.add_error(str(e))
                    
        # Validate additional validation paths
        for path_set in self._additional_val_paths:
            for path in path_set:
                if not path.file_path.exists():
                    result.add_error(f"Additional validation file does not exist: {path.file_path}")
                elif not path.genome_path:
                    try:
                        FASTAHandler.validate_format(path.file_path)
                    except GkmError as e:
                        result.add_error(str(e))
                        
        return result

