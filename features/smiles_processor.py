import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol
from rdkit import DataStructs
from transformers import AutoTokenizer
import torch
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import re
import hashlib
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SMILESProcessorConfig:
    """Configuration for SMILES processing."""
    max_length: int = 200
    tokenizer_name: str = "seyonec/PubChem10M_SMILES_BPE_396_250"
    use_chiral: bool = True
    use_isomeric: bool = True
    canonicalize: bool = True
    kekulize: bool = True
    remove_hydrogens: bool = False

@dataclass
class MolecularDescriptorConfig:
    """Configuration for molecular descriptor calculation."""
    include_fingerprints: bool = True
    include_2d_descriptors: bool = True
    include_3d_descriptors: bool = False
    fingerprint_type: str = "morgan"  # "morgan", "rdkit", "maccs"
    fingerprint_radius: int = 2
    fingerprint_size: int = 2048
    include_atom_counts: bool = True
    include_bond_counts: bool = True
    include_functional_groups: bool = True


class SMILESProcessor:
    """
    Advanced SMILES processor for chemical reaction data.
    
    This class handles SMILES string processing, tokenization, and molecular
    representation for machine learning applications.
    """
    
    def __init__(self, config: SMILESProcessorConfig = None):
        """
        Initialize the SMILES processor.
        
        Args:
            config: Configuration for SMILES processing
        """
        self.config = config or SMILESProcessorConfig()
        self.tokenizer = self._load_tokenizer()
        self.smiles_cache = {}
        
    def _load_tokenizer(self):
        """Load and configure the SMILES tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            
            # Ensure tokenizer has padding token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            logger.info(f"Loaded tokenizer: {self.config.tokenizer_name}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def preprocess_smiles(self, smiles: str) -> str:
        """
        Preprocess SMILES string for consistency.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            processed_smiles: Preprocessed SMILES string
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return ""
        
        try:
            # Remove extra whitespace
            smiles = smiles.strip()
            
            # Generate canonical SMILES if requested
            if self.config.canonicalize:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if self.config.kekulize:
                        try:
                            Chem.Kekulize(mol)
                        except:
                            pass  # Kekulization may fail for some structures
                    
                    canonical_smiles = Chem.MolToSmiles(
                        mol, 
                        isomericSmiles=self.config.use_isomeric,
                        kekuleSmiles=self.config.kekulize
                    )
                    return canonical_smiles
            
            return smiles
            
        except Exception as e:
            logger.warning(f"SMILES preprocessing failed for {smiles}: {e}")
            return smiles
    
    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Validate SMILES string and extract basic information.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            validation: Dictionary with validation results
        """
        result = {
            'valid': False,
            'canonical_smiles': '',
            'error': '',
            'num_atoms': 0,
            'num_heavy_atoms': 0,
            'molecular_weight': 0.0,
            'formula': ''
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                result['error'] = 'Invalid SMILES string'
                return result
            
            # Basic molecular properties
            result['valid'] = True
            result['num_atoms'] = mol.GetNumAtoms()
            result['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
            result['molecular_weight'] = Descriptors.MolWt(mol)
            result['formula'] = rdMolDescriptors.CalcMolFormula(mol)
            
            # Generate canonical SMILES
            result['canonical_smiles'] = Chem.MolToSmiles(
                mol, 
                isomericSmiles=self.config.use_isomeric
            )
            
            # Additional checks
            result['has_stereochemistry'] = any(
                atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED 
                for atom in mol.GetAtoms()
            )
            
            result['has_aromatic'] = any(
                bond.GetIsAromatic() for bond in mol.GetBonds()
            )
            
            result['ring_count'] = mol.GetRingInfo().NumRings()
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def tokenize_smiles(self, smiles: str, return_tensors: str = 'pt') -> Dict[str, torch.Tensor]:
        """
        Tokenize SMILES string for model input.
        
        Args:
            smiles: SMILES string to tokenize
            return_tensors: Format of returned tensors ('pt', 'np', None)
            
        Returns:
            tokenized: Dictionary with tokenized representation
        """
        # Use cache to avoid reprocessing
        cache_key = hashlib.md5(smiles.encode()).hexdigest()
        if cache_key in self.smiles_cache:
            return self.smiles_cache[cache_key]
        
        processed_smiles = self.preprocess_smiles(smiles)
        
        if not processed_smiles:
            # Return empty tokens for invalid SMILES
            empty_tokens = self.tokenizer(
                "",
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=return_tensors
            )
            self.smiles_cache[cache_key] = empty_tokens
            return empty_tokens
        
        try:
            tokenized = self.tokenizer(
                processed_smiles,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=return_tensors
            )
            
            self.smiles_cache[cache_key] = tokenized
            return tokenized
            
        except Exception as e:
            logger.warning(f"Tokenization failed for {smiles}: {e}")
            # Return empty tokens on failure
            empty_tokens = self.tokenizer(
                "",
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=return_tensors
            )
            self.smiles_cache[cache_key] = empty_tokens
            return empty_tokens
    
    def batch_tokenize(self, smiles_list: List[str], return_tensors: str = 'pt') -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            return_tensors: Format of returned tensors
            
        Returns:
            batch_tokens: Batch tokenized representation
        """
        processed_smiles = [self.preprocess_smiles(smiles) for smiles in smiles_list]
        valid_smiles = [s for s in processed_smiles if s]
        
        if not valid_smiles:
            # Return empty batch
            return self.tokenizer(
                [""] * len(smiles_list),
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=return_tensors
            )
        
        try:
            return self.tokenizer(
                valid_smiles,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors=return_tensors
            )
        except Exception as e:
            logger.error(f"Batch tokenization failed: {e}")
            # Fallback to individual tokenization
            all_input_ids = []
            all_attention_mask = []
            
            for smiles in smiles_list:
                tokenized = self.tokenize_smiles(smiles, return_tensors=None)
                all_input_ids.append(tokenized['input_ids'])
                all_attention_mask.append(tokenized['attention_mask'])
            
            return {
                'input_ids': torch.stack(all_input_ids) if return_tensors == 'pt' else all_input_ids,
                'attention_mask': torch.stack(all_attention_mask) if return_tensors == 'pt' else all_attention_mask
            }
    
    def smiles_to_embedding(self, smiles: str, model=None) -> Optional[torch.Tensor]:
        """
        Convert SMILES to embedding using a pre-trained model.
        
        Args:
            smiles: SMILES string
            model: Pre-trained model for embedding generation
            
        Returns:
            embedding: Molecular embedding vector
        """
        if model is None:
            logger.warning("No model provided for embedding generation")
            return None
        
        try:
            tokenized = self.tokenize_smiles(smiles, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model(**tokenized)
                # Use mean of last hidden states as embedding
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            return embedding.squeeze()
            
        except Exception as e:
            logger.error(f"Embedding generation failed for {smiles}: {e}")
            return None
    
    def similarity_search(self, query_smiles: str, candidate_smiles: List[str], 
                         metric: str = 'tanimoto') -> List[Tuple[str, float]]:
        """
        Find similar molecules based on fingerprint similarity.
        
        Args:
            query_smiles: Query SMILES string
            candidate_smiles: List of candidate SMILES strings
            metric: Similarity metric ('tanimoto', 'dice', 'cosine')
            
        Returns:
            similarities: List of (smiles, similarity) tuples sorted by similarity
        """
        try:
            query_mol = Chem.MolFromSmiles(query_smiles)
            if not query_mol:
                return []
            
            # Generate fingerprints
            query_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2).GetFingerprint(query_mol)
            
            similarities = []
            for candidate in candidate_smiles:
                candidate_mol = Chem.MolFromSmiles(candidate)
                if not candidate_mol:
                    continue
                
                candidate_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2).GetFingerprint(candidate_mol)
                
                if metric == 'tanimoto':
                    similarity = DataStructs.TanimotoSimilarity(query_fp, candidate_fp)
                elif metric == 'dice':
                    similarity = DataStructs.DiceSimilarity(query_fp, candidate_fp)
                elif metric == 'cosine':
                    similarity = DataStructs.CosineSimilarity(query_fp, candidate_fp)
                else:
                    similarity = DataStructs.TanimotoSimilarity(query_fp, candidate_fp)
                
                similarities.append((candidate, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def reaction_center_detection(self, reactant_smiles: str, product_smiles: str) -> Dict[str, Any]:
        """
        Detect reaction centers by comparing reactant and product.
        
        Args:
            reactant_smiles: Reactant SMILES string
            product_smiles: Product SMILES string
            
        Returns:
            reaction_info: Information about reaction centers
        """
        try:
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if not reactant_mol or not product_mol:
                return {'error': 'Invalid SMILES'}
            
            # Simple atom mapping (this is a simplified version)
            # In practice, more sophisticated algorithms should be used
            reactant_atoms = [atom.GetSymbol() for atom in reactant_mol.GetAtoms()]
            product_atoms = [atom.GetSymbol() for atom in product_mol.GetAtoms()]
            
            reaction_info = {
                'reactant_atom_count': len(reactant_atoms),
                'product_atom_count': len(product_atoms),
                'atom_count_change': len(product_atoms) - len(reactant_atoms),
                'bond_changes': self._detect_bond_changes(reactant_mol, product_mol),
                'functional_group_changes': self._detect_functional_group_changes(reactant_mol, product_mol)
            }
            
            return reaction_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_bond_changes(self, reactant_mol: Mol, product_mol: Mol) -> List[Dict]:
        """Detect bond changes between reactant and product."""
        # Simplified bond change detection
        # In practice, use proper reaction mapping algorithms
        changes = []
        
        reactant_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) 
                         for bond in reactant_mol.GetBonds()]
        product_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) 
                        for bond in product_mol.GetBonds()]
        
        # Find broken bonds
        for bond in reactant_bonds:
            if bond not in product_bonds:
                changes.append({
                    'type': 'bond_break',
                    'atoms': bond[:2],
                    'bond_type': str(bond[2])
                })
        
        # Find formed bonds
        for bond in product_bonds:
            if bond not in reactant_bonds:
                changes.append({
                    'type': 'bond_formation',
                    'atoms': bond[:2],
                    'bond_type': str(bond[2])
                })
        
        return changes
    
    def _detect_functional_group_changes(self, reactant_mol: Mol, product_mol: Mol) -> List[Dict]:
        """Detect functional group changes."""
        changes = []
        
        # Simple functional group detection
        functional_groups = {
            'alcohol': '[OX2H]',
            'carbonyl': 'C=O',
            'carboxylic_acid': 'C(=O)[OH]',
            'amine': '[NX3;H2,H1;!$(NC=O)]',
            'amide': 'C(=O)N',
            'ester': 'C(=O)O',
            'alkene': 'C=C',
            'alkyne': 'C#C'
        }
        
        for group, smarts in functional_groups.items():
            reactant_count = len(reactant_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
            product_count = len(product_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
            
            if reactant_count != product_count:
                changes.append({
                    'functional_group': group,
                    'reactant_count': reactant_count,
                    'product_count': product_count,
                    'change': product_count - reactant_count
                })
        
        return changes
    
    def visualize_molecules(self, smiles_list: List[str], save_path: str = None, 
                          labels: List[str] = None, mols_per_row: int = 4):
        """
        Visualize multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings to visualize
            save_path: Path to save the image (optional)
            labels: Labels for each molecule (optional)
            mols_per_row: Number of molecules per row in the grid
        """
        try:
            mols = []
            valid_labels = []
            
            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
                    if labels and i < len(labels):
                        valid_labels.append(labels[i])
                    else:
                        valid_labels.append(f"Mol_{i+1}")
            
            if not mols:
                logger.warning("No valid molecules to visualize")
                return
            
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=mols_per_row,
                subImgSize=(300, 300),
                legends=valid_labels,
                returnPNG=False
            )
            
            if save_path:
                img.save(save_path)
                logger.info(f"Molecules saved to {save_path}")
            
            return img
            
        except Exception as e:
            logger.error(f"Molecular visualization failed: {e}")
            return None


class MolecularFeatureExtractor:
    """
    Comprehensive molecular feature extractor for machine learning.
    
    This class calculates various molecular descriptors and fingerprints
    that can be used as features in machine learning models.
    """
    
    def __init__(self, config: MolecularDescriptorConfig = None):
        """
        Initialize the molecular feature extractor.
        
        Args:
            config: Configuration for feature extraction
        """
        self.config = config or MolecularDescriptorConfig()
        self.feature_labels = self._initialize_feature_labels()
        
    def _initialize_feature_labels(self) -> List[str]:
        """Initialize comprehensive list of feature labels."""
        labels = []
        
        # Basic molecular descriptors
        basic_descriptors = [
            'MolWt', 'HeavyAtomMolWt', 'NumAtoms', 'NumHeavyAtoms',
            'NumRotatableBonds', 'NumHDonors', 'NumHAcceptors',
            'TPSA', 'LabuteASA', 'MolLogP', 'MolMR'
        ]
        
        if self.config.include_atom_counts:
            atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            for atom in atom_types:
                labels.append(f'Num{atom}Atoms')
        
        if self.config.include_bond_counts:
            bond_types = ['Single', 'Double', 'Triple', 'Aromatic']
            for bond in bond_types:
                labels.append(f'Num{bond}Bonds')
        
        labels.extend(basic_descriptors)
        
        # Fingerprint features
        if self.config.include_fingerprints:
            fp_size = self.config.fingerprint_size
            labels.extend([f'FP_{i}' for i in range(fp_size)])
        
        return labels
    
    def get_2d_descriptors(self, mol: Mol) -> Dict[str, float]:
        """
        Calculate 2D molecular descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            descriptors: Dictionary of 2D molecular descriptors
        """
        descriptors = {}
        
        try:
            # Basic counts
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
            descriptors['NumAtoms'] = mol.GetNumAtoms()
            descriptors['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            
            # Surface area and polarity
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
            
            # Lipophilicity and molar refractivity
            descriptors['MolLogP'] = Descriptors.MolLogP(mol)
            descriptors['MolMR'] = Descriptors.MolMR(mol)
            
            # Atom type counts
            if self.config.include_atom_counts:
                atom_counts = defaultdict(int)
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    atom_counts[symbol] += 1
                
                for atom_type in ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']:
                    descriptors[f'Num{atom_type}Atoms'] = atom_counts.get(atom_type, 0)
            
            # Bond type counts
            if self.config.include_bond_counts:
                bond_counts = defaultdict(int)
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    bond_counts[str(bond_type)] += 1
                
                for bond_type in ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']:
                    key = bond_type.capitalize()
                    descriptors[f'Num{key}Bonds'] = bond_counts.get(bond_type, 0)
            
            # Ring information
            descriptors['NumRings'] = mol.GetRingInfo().NumRings()
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            
            # Charge-related descriptors
            descriptors['FractionCSP3'] = Descriptors.FractionCSP3(mol)
            descriptors['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)
            descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
            
        except Exception as e:
            logger.warning(f"2D descriptor calculation failed: {e}")
            # Set default values for failed calculations
            for label in self._initialize_feature_labels():
                if not label.startswith('FP_'):
                    descriptors[label] = 0.0
        
        return descriptors
    
    def get_fingerprint(self, mol: Mol) -> np.ndarray:
        """
        Generate molecular fingerprint.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            fingerprint: Molecular fingerprint as numpy array
        """
        try:
            if self.config.fingerprint_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 
                    radius=self.config.fingerprint_radius,
                    nBits=self.config.fingerprint_size
                )
            elif self.config.fingerprint_type == "rdkit":
                fp = Chem.RDKFingerprint(mol, fpSize=self.config.fingerprint_size)
            elif self.config.fingerprint_type == "maccs":
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 
                    radius=2,
                    nBits=self.config.fingerprint_size
                )
            
            # Convert to numpy array
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
            
        except Exception as e:
            logger.warning(f"Fingerprint generation failed: {e}")
            return np.zeros(self.config.fingerprint_size)
    
    def extract_features(self, smiles: str) -> np.ndarray:
        """
        Extract comprehensive molecular features.
        
        Args:
            smiles: SMILES string
            
        Returns:
            features: Combined feature vector
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return np.zeros(len(self.feature_labels))
            
            features = []
            
            # 2D descriptors
            if self.config.include_2d_descriptors:
                descriptors = self.get_2d_descriptors(mol)
                for label in self.feature_labels:
                    if not label.startswith('FP_'):
                        features.append(descriptors.get(label, 0.0))
            
            # Fingerprints
            if self.config.include_fingerprints:
                fp = self.get_fingerprint(mol)
                features.extend(fp.tolist())
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {smiles}: {e}")
            return np.zeros(len(self.feature_labels))
    
    def batch_extract_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        Extract features for a batch of molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            features: 2D array of features [n_molecules, n_features]
        """
        features = []
        valid_smiles = []
        
        for smiles in smiles_list:
            feature_vector = self.extract_features(smiles)
            if not np.all(feature_vector == 0):  # Skip completely zero vectors
                features.append(feature_vector)
                valid_smiles.append(smiles)
        
        if not features:
            return np.zeros((0, len(self.feature_labels)))
        
        return np.array(features), valid_smiles


# Utility functions
def create_smiles_processor(processor_type: str = 'standard', **kwargs):
    """
    Factory function to create SMILES processor instances.
    
    Args:
        processor_type: Type of processor
        **kwargs: Configuration parameters
        
    Returns:
        processor: SMILES processor instance
    """
    processors = {
        'standard': SMILESProcessor,
        'feature_extractor': MolecularFeatureExtractor
    }
    
    if processor_type not in processors:
        raise ValueError(f"Unknown processor type: {processor_type}. Available: {list(processors.keys())}")
    
    processor_class = processors[processor_type]
    return processor_class(**kwargs)


def validate_reaction_smiles(reactant_smiles: str, product_smiles: str) -> Dict[str, Any]:
    """
    Validate reaction SMILES pair for consistency.
    
    Args:
        reactant_smiles: Reactant SMILES
        product_smiles: Product SMILES
        
    Returns:
        validation: Validation results
    """
    processor = SMILESProcessor()
    
    reactant_validation = processor.validate_smiles(reactant_smiles)
    product_validation = processor.validate_smiles(product_smiles)
    
    validation = {
        'reactant': reactant_validation,
        'product': product_validation,
        'reaction_valid': reactant_validation['valid'] and product_validation['valid']
    }
    
    if validation['reaction_valid']:
        # Check for atom conservation (simplified)
        atom_diff = product_validation['num_heavy_atoms'] - reactant_validation['num_heavy_atoms']
        validation['atom_conservation'] = atom_diff == 0
        validation['atom_difference'] = atom_diff
        
        # Detect reaction centers
        reaction_centers = processor.reaction_center_detection(reactant_smiles, product_smiles)
        validation['reaction_centers'] = reaction_centers
    
    return validation


# Example usage and testing
if __name__ == "__main__":
    # Test SMILES processor
    print("Testing SMILESProcessor...")
    processor = SMILESProcessor()
    
    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
        "invalid_smiles"  # Should fail
    ]
    
    for smiles in test_smiles:
        print(f"\nProcessing: {smiles}")
        validation = processor.validate_smiles(smiles)
        print(f"Validation: {validation}")
        
        if validation['valid']:
            tokenized = processor.tokenize_smiles(smiles)
            print(f"Tokenized shape: {tokenized['input_ids'].shape}")
    
    # Test molecular feature extractor
    print("\nTesting MolecularFeatureExtractor...")
    feature_extractor = MolecularFeatureExtractor()
    
    for smiles in test_smiles[:3]:  # Skip invalid
        features = feature_extractor.extract_features(smiles)
        print(f"{smiles}: Features shape: {features.shape}, Non-zero: {np.count_nonzero(features)}")
    
    # Test reaction validation
    print("\nTesting reaction validation...")
    reaction_validation = validate_reaction_smiles("CCO", "CCOC(=O)O")
    print(f"Reaction validation: {reaction_validation}")
    
    print("\nAll SMILES processing tests completed successfully!")
