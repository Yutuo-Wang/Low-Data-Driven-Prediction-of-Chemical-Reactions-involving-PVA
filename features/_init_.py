"""
PVA-ReAct Feature Engineering
=============================

This package contains feature extraction and processing modules for chemical reactions.

Available Modules:
- DFTFeatureExtractor: Quantum chemical feature extraction using DFT calculations
- SMILESProcessor: SMILES string processing and molecular representation
"""

from .dft_extractor import EnhancedDFTFeatureExtractor, DFTFeatureExtractor
from .smiles_processor import SMILESProcessor, MolecularFeatureExtractor

__all__ = [
    'EnhancedDFTFeatureExtractor',
    'DFTFeatureExtractor', 
    'SMILESProcessor',
    'MolecularFeatureExtractor'
]

__version__ = '1.0.0'
