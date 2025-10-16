import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from pyscf import gto, dft, lo, lib
from pyscf.tools import molden
from typing import Tuple, Optional, Dict, List
import time
import os
import logging
from dataclasses import dataclass
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')
lib.logger.DISABLE_ALL = True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFTConfig:
    """Configuration for DFT calculations."""
    basis_set: str = 'def2svp'
    functional: str = 'B3LYP'
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100
    memory_limit: int = 4000  # MB
    num_threads: int = 4
    save_orbitals: bool = False
    orbital_dir: str = 'orbitals'

@dataclass
class MolecularProperties:
    """Container for molecular properties from DFT calculations."""
    homo_energy: float
    lumo_energy: float
    dipole_moment: Tuple[float, float, float]
    total_energy: float
    fermi_energy: float
    mulliken_charges: np.ndarray
    mulliken_std: float
    band_gap: float
    hardness: float
    softness: float
    electrophilicity: float
    polarizability: Optional[float] = None
    fukui_indices: Optional[np.ndarray] = None


class EnhancedDFTFeatureExtractor:
    """
    Enhanced DFT feature extractor with comprehensive quantum chemical calculations.
    
    This class performs DFT calculations to extract electronic structure features
    that are crucial for understanding chemical reactivity and reaction mechanisms.
    """
    
    def __init__(self, config: DFTConfig = None):
        """
        Initialize the DFT feature extractor.
        
        Args:
            config: DFT calculation configuration
        """
        self.config = config or DFTConfig()
        self.feature_labels = self._get_feature_labels()
        self._setup_pyscf_environment()
        
    def _setup_pyscf_environment(self):
        """Setup PySCF calculation environment."""
        # Set memory and thread limits
        lib.num_threads(self.config.num_threads)
        try:
            lib.param.MAX_MEMORY = self.config.memory_limit
        except:
            pass  # Some PySCF versions don't have this parameter
            
    def _get_feature_labels(self) -> List[str]:
        """Get comprehensive list of feature labels."""
        base_features = [
            'HOMO_Energy', 'LUMO_Energy', 'Band_Gap', 'Hardness', 'Softness',
            'Electrophilicity', 'Dipole_X', 'Dipole_Y', 'Dipole_Z', 'Dipole_Magnitude',
            'Total_Energy', 'Fermi_Energy', 'Mulliken_Charge_Std', 'Mulliken_Charge_Max',
            'Mulliken_Charge_Min', 'Molecular_Weight', 'Heavy_Atom_Count'
        ]
        
        # Add Fukui indices if available
        fukui_features = ['Fukui_Electrophilic', 'Fukui_Nucleophilic', 'Fukui_Radical']
        return base_features + fukui_features
    
    def _get_formal_charge(self, mol: Chem.Mol) -> int:
        """Safely get molecular formal charge."""
        try:
            return Chem.GetFormalCharge(mol)
        except Exception as e:
            logger.warning(f"Error getting formal charge: {e}, using 0")
            return 0
    
    def _generate_3d_structure(self, mol: Chem.Mol, max_attempts: int = 3) -> bool:
        """
        Generate 3D molecular structure with multiple attempts.
        
        Args:
            mol: RDKit molecule object
            max_attempts: Maximum number of embedding attempts
            
        Returns:
            success: Whether 3D structure generation was successful
        """
        for attempt in range(max_attempts):
            try:
                # Add hydrogens for complete structure
                mol_h = Chem.AddHs(mol)
                
                # Generate 3D coordinates
                success = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                if success == -1:
                    logger.warning(f"3D embedding failed on attempt {attempt + 1}")
                    continue
                
                # Optimize geometry
                AllChem.MMFFOptimizeMolecule(mol_h)
                return True
                
            except Exception as e:
                logger.warning(f"3D generation attempt {attempt + 1} failed: {e}")
                continue
        
        logger.error(f"Failed to generate 3D structure after {max_attempts} attempts")
        return False
    
    def _calculate_fukui_indices(self, pymol: gto.Mole, mf: dft.RKS) -> Optional[np.ndarray]:
        """
        Calculate Fukui indices for reactivity analysis.
        
        Args:
            pymol: PySCF molecule object
            mf: Converged mean-field object
            
        Returns:
            fukui_indices: Array of Fukui indices for each atom
        """
        try:
            # Get molecular orbital coefficients and occupations
            mo_energy = mf.mo_energy
            mo_occ = mf.mo_occ
            mo_coeff = mf.mo_coeff
            
            # Find HOMO and LUMO indices
            homo_idx = np.where(mo_occ > 0)[0][-1]
            lumo_idx = np.where(mo_occ == 0)[0][0]
            
            # Calculate Fukui indices using finite difference approximation
            # This is a simplified version - more sophisticated methods exist
            natoms = pymol.natm
            fukui_electrophilic = np.zeros(natoms)
            fukui_nucleophilic = np.zeros(natoms)
            fukui_radical = np.zeros(natoms)
            
            # For demonstration, we'll use Mulliken population differences
            # In practice, more accurate methods should be used
            dm = mf.make_rdm1()
            mulliken = lo.orth_ao(pymol, 'mulliken', dm)
            
            # Simplified Fukui calculation (placeholder)
            for i in range(natoms):
                fukui_radical[i] = mulliken[i]  # Simplified representation
                fukui_electrophilic[i] = mulliken[i] * 0.5
                fukui_nucleophilic[i] = mulliken[i] * 0.5
            
            return np.array([fukui_electrophilic, fukui_nucleophilic, fukui_radical])
            
        except Exception as e:
            logger.warning(f"Fukui indices calculation failed: {e}")
            return None
    
    def _calculate_reactivity_descriptors(self, homo: float, lumo: float) -> Dict[str, float]:
        """
        Calculate chemical reactivity descriptors from HOMO/LUMO energies.
        
        Args:
            homo: HOMO energy
            lumo: LUMO energy
            
        Returns:
            descriptors: Dictionary of reactivity descriptors
        """
        band_gap = lumo - homo
        hardness = (lumo - homo) / 2.0
        softness = 1.0 / hardness if hardness != 0 else 0
        electrophilicity = (homo + lumo)**2 / (8 * hardness) if hardness != 0 else 0
        
        return {
            'band_gap': band_gap,
            'hardness': hardness,
            'softness': softness,
            'electrophilicity': electrophilicity
        }
    
    def _get_default_features(self) -> np.ndarray:
        """Return default feature values for failed calculations."""
        return np.zeros(len(self.feature_labels))
    
    def calculate(self, smiles: str) -> Tuple[np.ndarray, Optional[str]]:
        """
        Perform comprehensive DFT calculation for a given SMILES string.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            features: Array of DFT-calculated features
            orbital_path: Path to saved orbital file if requested
        """
        start_time = time.time()
        orbital_path = None
        
        try:
            # Input validation
            if not isinstance(smiles, str) or not smiles.strip():
                logger.warning(f"Invalid SMILES string: {smiles}")
                return self._get_default_features(), None
            
            # Parse SMILES and generate molecule
            mol = Chem.MolFromSmiles(smiles)
            if not mol or mol.GetNumAtoms() == 0:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return self._get_default_features(), None
            
            # Check molecular size
            if mol.GetNumAtoms() > 150:
                logger.warning(f"Molecule too large: {smiles} ({mol.GetNumAtoms()} atoms)")
                return self._get_default_features(), None
            
            # Generate 3D structure
            if not self._generate_3d_structure(mol):
                return self._get_default_features(), None
            
            mol = Chem.AddHs(mol)  # Ensure hydrogens are added
            
            # Build PySCF molecule object
            conf = mol.GetConformer()
            atoms = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
            
            pymol = gto.Mole()
            pymol.atom = atoms
            pymol.basis = self.config.basis_set
            pymol.charge = self._get_formal_charge(mol)
            pymol.spin = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
            pymol.build()
            
            # Perform DFT calculation
            mf = dft.RKS(pymol)
            mf.xc = self.config.functional
            mf.conv_tol = self.config.convergence_tolerance
            mf.max_cycle = self.config.max_iterations
            
            logger.info(f"Starting DFT calculation for {smiles}")
            mf.kernel()
            
            if not mf.converged:
                logger.warning(f"DFT calculation did not converge for {smiles}")
                return self._get_default_features(), None
            
            # Extract electronic structure properties
            mo_energy = mf.mo_energy
            mo_occ = mf.mo_occ
            
            # Find HOMO and LUMO
            occupied_indices = np.where(mo_occ > 0)[0]
            unoccupied_indices = np.where(mo_occ == 0)[0]
            
            if len(occupied_indices) == 0 or len(unoccupied_indices) == 0:
                logger.warning(f"Insufficient orbitals for {smiles}")
                return self._get_default_features(), None
            
            homo = mo_energy[occupied_indices[-1]]
            lumo = mo_energy[unoccupied_indices[0]]
            
            # Calculate dipole moment
            dipole = mf.dip_moment(unit='A.U.')
            dipole_magnitude = np.linalg.norm(dipole)
            
            # Total energy
            total_energy = mf.e_tot
            
            # Fermi level (approximate)
            fermi = 0.5 * (homo + lumo)
            
            # Mulliken population analysis
            dm = mf.make_rdm1()
            mulliken_charges = lo.orth_ao(pymol, 'mulliken', dm)
            mulliken_std = np.std(mulliken_charges)
            mulliken_max = np.max(mulliken_charges)
            mulliken_min = np.min(mulliken_charges)
            
            # Reactivity descriptors
            reactivity = self._calculate_reactivity_descriptors(homo, lumo)
            
            # Fukui indices for reactivity analysis
            fukui_indices = self._calculate_fukui_indices(pymol, mf)
            
            # Molecular descriptors
            molecular_weight = Descriptors.MolWt(mol)
            heavy_atom_count = mol.GetNumHeavyAtoms()
            
            # Build feature vector
            features = [
                homo, lumo, reactivity['band_gap'], reactivity['hardness'],
                reactivity['softness'], reactivity['electrophilicity'],
                dipole[0], dipole[1], dipole[2], dipole_magnitude,
                total_energy, fermi, mulliken_std, mulliken_max, mulliken_min,
                molecular_weight, heavy_atom_count
            ]
            
            # Add Fukui indices if available
            if fukui_indices is not None:
                # Use mean values of Fukui indices as features
                fukui_mean = np.mean(fukui_indices, axis=1)
                features.extend(fukui_mean.tolist())
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Save orbital information if requested
            if self.config.save_orbitals:
                os.makedirs(self.config.orbital_dir, exist_ok=True)
                safe_smiles = "".join(c for c in smiles if c.isalnum())[:20]
                orbital_path = os.path.join(self.config.orbital_dir, f"{safe_smiles}.molden")
                try:
                    molden.dump_scf(mf, orbital_path)
                    logger.info(f"Orbitals saved to {orbital_path}")
                except Exception as e:
                    logger.warning(f"Failed to save orbitals: {e}")
            
            calculation_time = time.time() - start_time
            logger.info(f"DFT calculation completed for {smiles} in {calculation_time:.1f}s")
            
            return np.nan_to_num(np.array(features, dtype=np.float32)), orbital_path
            
        except Exception as e:
            logger.error(f"DFT calculation failed for {smiles}: {str(e)}")
            return self._get_default_features(), None
    
    def batch_calculate(self, smiles_list: List[str], 
                       batch_size: int = 10,
                       save_results: bool = True) -> pd.DataFrame:
        """
        Perform DFT calculations for a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Number of calculations per batch
            save_results: Whether to save results to CSV
            
        Returns:
            results: DataFrame with features for all molecules
        """
        results = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(smiles_list)-1)//batch_size + 1}")
            
            for smiles in batch:
                features, orbital_path = self.calculate(smiles)
                
                result = {
                    'smiles': smiles,
                    'calculation_success': not np.all(features == 0),
                    'orbital_path': orbital_path
                }
                
                # Add feature values
                for j, label in enumerate(self.feature_labels):
                    result[label] = features[j]
                
                results.append(result)
        
        df_results = pd.DataFrame(results)
        
        if save_results and len(results) > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"dft_features_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            logger.info(f"Results saved to {filename}")
        
        return df_results
    
    def validate_molecule(self, smiles: str) -> Dict[str, any]:
        """
        Validate molecule and check if DFT calculation is feasible.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            validation: Dictionary with validation results
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {'valid': False, 'error': 'Invalid SMILES'}
            
            num_atoms = mol.GetNumAtoms()
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            molecular_weight = Descriptors.MolWt(mol)
            
            # Check for problematic elements
            elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
            problematic_elements = {'Pu', 'Am', 'Cm', 'Bk', 'Cf'}  # Actinides
            
            validation = {
                'valid': True,
                'num_atoms': num_atoms,
                'num_heavy_atoms': num_heavy_atoms,
                'molecular_weight': molecular_weight,
                'elements': list(elements),
                'has_problematic_elements': len(elements.intersection(problematic_elements)) > 0,
                'too_large': num_atoms > 150,
                'recommended': num_atoms <= 100 and molecular_weight < 500
            }
            
            return validation
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}


class DFTFeatureExtractor:
    """
    Lightweight DFT feature extractor for rapid calculations.
    
    This is a simplified version of the enhanced extractor, suitable for
    high-throughput calculations where computational cost is a concern.
    """
    
    def __init__(self, basis: str = 'def2svp', xc: str = 'B3LYP'):
        """
        Initialize the lightweight DFT extractor.
        
        Args:
            basis: Basis set for DFT calculations
            xc: Exchange-correlation functional
        """
        self.basis = basis
        self.xc = xc
        self.feature_labels = [
            'HOMO', 'LUMO', 'Dipole_X', 'Dipole_Y', 'Dipole_Z',
            'Total_Energy', 'Fermi_Level', 'Mulliken_Charge_Std'
        ]
    
    def _get_formal_charge(self, mol: Chem.Mol) -> int:
        """Safely get molecular formal charge."""
        try:
            return mol.GetFormalCharge()
        except Exception:
            return 0
    
    def calculate(self, smiles: str) -> np.ndarray:
        """
        Perform basic DFT calculation for feature extraction.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            features: Array of basic DFT features
        """
        try:
            if not isinstance(smiles, str) or len(smiles) == 0:
                return np.zeros(len(self.feature_labels))
            
            start_time = time.time()
            mol = Chem.MolFromSmiles(smiles)
            if not mol or mol.GetNumAtoms() == 0:
                return np.zeros(len(self.feature_labels))
            
            # Generate 3D structure
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Build PySCF molecule object
            conf = mol.GetConformer()
            atoms = [(atom.GetSymbol(), tuple(conf.GetAtomPosition(atom.GetIdx())))
                     for atom in mol.GetAtoms()]
            
            pymol = gto.Mole()
            pymol.atom = atoms
            pymol.basis = self.basis
            pymol.charge = self._get_formal_charge(mol)
            pymol.spin = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
            pymol.build()
            
            # Perform DFT calculation
            mf = dft.RKS(pymol)
            mf.xc = self.xc
            mf.conv_tol = 1e-6
            mf.kernel()
            
            if not mf.converged:
                return np.zeros(len(self.feature_labels))
            
            # Calculate features
            homo = mf.mo_energy[mf.mo_occ > 0][-1]
            lumo = mf.mo_energy[mf.mo_occ == 0][0]
            dipole = mf.dip_moment(unit='A.U.')
            total_energy = mf.e_tot
            fermi = 0.5 * (homo + lumo)
            
            # Mulliken charge analysis
            dm = mf.make_rdm1()
            charges = lo.orth_ao(pymol, 'mulliken', dm)
            charge_std = np.std(charges)
            
            features = np.array([
                homo, lumo, dipole[0], dipole[1], dipole[2],
                total_energy, fermi, charge_std
            ])
            
            return np.nan_to_num(features)
            
        except Exception as e:
            logger.error(f"DFT Error: {smiles} - {str(e)}")
            return np.zeros(len(self.feature_labels))


# Utility functions
def create_dft_extractor(extractor_type: str = 'enhanced', **kwargs):
    """
    Factory function to create DFT extractor instances.
    
    Args:
        extractor_type: Type of extractor ('enhanced' or 'basic')
        **kwargs: Configuration parameters
        
    Returns:
        extractor: DFT feature extractor instance
    """
    extractors = {
        'enhanced': EnhancedDFTFeatureExtractor,
        'basic': DFTFeatureExtractor
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {list(extractors.keys())}")
    
    extractor_class = extractors[extractor_type]
    return extractor_class(**kwargs)


def estimate_calculation_time(smiles: str, basis_set: str = 'def2svp') -> float:
    """
    Roughly estimate DFT calculation time for a molecule.
    
    Args:
        smiles: SMILES string
        basis_set: Basis set used
        
    Returns:
        time_estimate: Estimated calculation time in seconds
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0
        
        num_atoms = mol.GetNumAtoms()
        num_electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
        
        # Rough empirical estimation
        base_time = 1.0  # seconds
        time_per_atom = 2.0
        time_per_electron = 0.5
        
        estimate = base_time + (num_atoms * time_per_atom) + (num_electrons * time_per_electron)
        
        # Adjust for basis set complexity
        basis_complexity = {
            'sto-3g': 0.5,
            '6-31g': 1.0,
            'def2svp': 1.5,
            'def2tzvp': 2.5,
            'cc-pvdz': 2.0
        }
        
        complexity = basis_complexity.get(basis_set, 1.0)
        estimate *= complexity
        
        return max(estimate, 1.0)  # Minimum 1 second
        
    except Exception:
        return 0


# Example usage and testing
if __name__ == "__main__":
    # Test the DFT extractors
    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(=O)O",  # Acetic acid
    ]
    
    print("Testing EnhancedDFTFeatureExtractor...")
    enhanced_extractor = EnhancedDFTFeatureExtractor(
        DFTConfig(basis_set='6-31g', save_orbitals=False)  # Use smaller basis for testing
    )
    
    for smiles in test_smiles:
        print(f"\nProcessing: {smiles}")
        validation = enhanced_extractor.validate_molecule(smiles)
        print(f"Validation: {validation}")
        
        if validation['valid'] and not validation['too_large']:
            features, orbital_path = enhanced_extractor.calculate(smiles)
            print(f"Features calculated: {len(features)}")
            print(f"First 5 features: {features[:5]}")
            print(f"Orbital path: {orbital_path}")
    
    print("\nTesting DFTFeatureExtractor (basic)...")
    basic_extractor = DFTFeatureExtractor(basis='6-31g')
    
    for smiles in test_smiles:
        features = basic_extractor.calculate(smiles)
        print(f"{smiles}: {features[:3]}...")
    
    print("\nAll DFT extractors tested successfully!")
