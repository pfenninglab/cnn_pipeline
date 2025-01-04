"""gkm_utils.py: Utility functions for handling FASTA files in gkm-SVM pipeline.

This module provides file handling and validation utilities for gkm-SVM pipeline, 
focusing on:
1. FASTA file validation and preparation 
2. Path resolution matching CNN pipeline
3. BED to FASTA conversion
4. Input sequence validation
"""

import os
from typing import List, Dict, Optional, Union
from pathlib import Path
from gkm_config import GkmConfig

def resolve_paths_with_config(config: GkmConfig) -> None:
    """Resolve all paths in a GkmConfig object.
    
    Args:
        config: GkmConfig object containing paths to resolve
    """
    config.resolve_paths()

def validate_config_paths(config: GkmConfig) -> None:
    """Validate all paths in a GkmConfig object.
    
    Args:
        config: GkmConfig object containing paths to validate
    """
    config._validate_resolved_paths()

class GkmUtilsError(Exception):
    """Base exception class for gkm-utils errors."""
    pass

class PathValidator:
    """Validates file paths and formats matching CNN pipeline."""
    
    @staticmethod
    def validate_input_file(filepath: Union[str, Path]) -> None:
        """Check if input file exists and is readable."""
        path = Path(filepath)
        if not path.exists():
            raise GkmUtilsError(f"File does not exist: {filepath}")
        if not os.access(path, os.R_OK):
            raise GkmUtilsError(f"File is not readable: {filepath}")
            
    @staticmethod
    def validate_output_path(filepath: Union[str, Path]) -> None:
        """Validate output path is writable."""
        path = Path(filepath)
        if path.exists() and not os.access(path, os.W_OK):
            raise GkmUtilsError(f"Path exists but is not writable: {filepath}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise GkmUtilsError(f"Cannot create parent directories for {filepath}: {e}")

    @staticmethod
    def validate_fasta_format(filepath: Union[str, Path]) -> None:
        """Validate FASTA file format."""
        with open(filepath) as f:
            first_line = f.readline().strip()
            if not first_line.startswith('>'):
                raise GkmUtilsError(f"Invalid FASTA format in {filepath}")

class SequenceValidator:
    """Validates sequences for gkm-SVM processing."""

    @staticmethod
    def validate_sequence_length(filepath: Union[str, Path]) -> None:
        """Validate all sequences have same length."""
        lengths = set()
        current_seq = []
        
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        lengths.add(len(''.join(current_seq)))
                        current_seq = []
                else:
                    current_seq.append(line)
                    
        if current_seq:
            lengths.add(len(''.join(current_seq)))
            
        if len(lengths) > 1:
            raise GkmUtilsError(f"Sequences have different lengths in {filepath}: {lengths}")
            
        if not lengths:
            raise GkmUtilsError(f"No sequences found in {filepath}")

class FASTAHandler:
    """Handles FASTA file operations for gkm-SVM pipeline."""
    
    @staticmethod
    def bed_to_fasta(bed_file: Union[str, Path], 
                     genome_file: Union[str, Path],
                     output_path: Union[str, Path]) -> None:
        """Convert BED file to FASTA using bedtools."""
        try:
            import pybedtools
        except ImportError:
            raise GkmUtilsError("pybedtools required for BED to FASTA conversion")

        PathValidator.validate_input_file(bed_file)
        PathValidator.validate_input_file(genome_file)
        PathValidator.validate_output_path(output_path)

        try:
            bed = pybedtools.BedTool(bed_file)
            bed.sequence(fi=str(genome_file), fo=str(output_path))
            
            # Validate output
            if not os.path.exists(output_path):
                raise GkmUtilsError(f"FASTA file not created: {output_path}")
                
            SequenceValidator.validate_sequence_length(output_path)
            
        except Exception as e:
            raise GkmUtilsError(f"BED to FASTA conversion failed: {str(e)}")

def resolve_input_files(config_files: List[Union[str, Dict]],
                       genome: Optional[str] = None) -> List[str]:
    """Resolve input files to FASTA format matching CNN pipeline.
    
    Args:
        config_files: List of file paths or dicts from config
        genome: Optional genome path for BED conversion
        
    Returns:
        List of paths to FASTA files
    """
    fasta_files = []
    
    for file_entry in config_files:
        if isinstance(file_entry, dict):
            # Handle dictionary format from CNN pipeline config
            if 'genome' in file_entry and 'intervals' in file_entry:
                # BED/narrowPeak with reference genome
                PathValidator.validate_input_file(file_entry['genome'])
                PathValidator.validate_input_file(file_entry['intervals'])
                
                fasta_path = str(Path(file_entry['intervals']).with_suffix('.fa'))
                FASTAHandler.bed_to_fasta(
                    file_entry['intervals'],
                    file_entry['genome'],
                    fasta_path
                )
                fasta_files.append(fasta_path)
            else:
                # Direct FASTA path
                PathValidator.validate_input_file(file_entry['path'])
                PathValidator.validate_fasta_format(file_entry['path'])
                fasta_files.append(file_entry['path'])
        else:
            # Handle direct file path
            if genome and str(file_entry).endswith(('.bed', '.narrowPeak')):
                # Convert BED to FASTA
                PathValidator.validate_input_file(genome)
                PathValidator.validate_input_file(file_entry)
                
                fasta_path = str(Path(file_entry).with_suffix('.fa'))
                FASTAHandler.bed_to_fasta(file_entry, genome, fasta_path)
                fasta_files.append(fasta_path)
            else:
                # Direct FASTA file
                PathValidator.validate_input_file(file_entry)
                PathValidator.validate_fasta_format(file_entry)
                fasta_files.append(str(file_entry))
                
    # Validate all sequences have same length
    for fasta_file in fasta_files:
        SequenceValidator.validate_sequence_length(fasta_file)
        
    return fasta_files