import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft, lo, lib
from pyscf.tools import molden

class DFTFeatureExtractor:
    def __init__(self, basis='def2svp', xc='B3LYP', save_orbital=False):
        self.basis = basis
        self.xc = xc
        self.save_orbital = save_orbital
        self.feature_labels = [
            'HOMO', 'LUMO', 'Dipole_X', 'Dipole_Y', 'Dipole_Z',
            'Total_Energy', 'Fermi_Level', 'Mulliken_Charge_Std'
        ]
        
    def _get_formal_charge(self, mol):
        """Safely retrieve the formal charge of the molecule"""
        try:
            return mol.GetFormalCharge()
        except AttributeError:
            return 0

    def _calculate_charges(self, pymol):
        """Calculate Mulliken charges"""
        mf = dft.RKS(pymol)
        mf.xc = self.xc
        mf.kernel()
        dm = mf.make_rdm1()
        mulliken = lo.orth_ao(pymol, 'mulliken', dm)
        return mulliken

    def calculate(self, smiles):
        """Enhanced DFT feature calculation"""
        try:
            if not isinstance(smiles, str) or len(smiles) == 0:
                raise ValueError("Invalid SMILES string")
            
            start_time = time.time()
            mol = Chem.MolFromSmiles(smiles)
            if not mol or mol.GetNumAtoms() == 0:
                return np.zeros(len(self.feature_labels)), None
            
            # Generate 3D structure
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Build PySCF molecule object
            conf = mol.GetConformer()
            atoms = [(atom.GetSymbol(), tuple(conf.GetAtomPosition(atom.GetIdx())) )
                     for atom in mol.GetAtoms()]
            
            pymol = gto.Mole()
            pymol.atom = atoms
            pymol.basis = self.basis
            pymol.charge = self._get_formal_charge(mol)
            pymol.spin = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())  # Adjust this line
            pymol.build()
            
            # Perform DFT calculation
            mf = dft.RKS(pymol)
            mf.xc = self.xc
            mf.conv_tol = 1e-6
            mf.kernel()
            
            if not mf.converged:
                return np.zeros(len(self.feature_labels)), None
            
            # Calculate features
            homo = mf.mo_energy[mf.mo_occ > 0][-1]
            lumo = mf.mo_energy[mf.mo_occ == 0][0]
            dipole = mf.dip_moment(unit='A.U.')
            total_energy = mf.e_tot
            fermi = 0.5 * (homo + lumo)
            
            # Mulliken charge analysis
            charges = self._calculate_charges(pymol)
            charge_std = np.std(charges)
            
            # Save orbital information
            orbital_path = None
            if self.save_orbital:
                os.makedirs("orbitals", exist_ok=True)
                orbital_path = f"orbitals/{smiles[:10]}.molden"
                molden.dump_scf(mf, orbital_path)
            
            features = np.array([
                homo, lumo, dipole[0], dipole[1], dipole[2],
                total_energy, fermi, charge_std
            ])
            
            time_cost = time.time() - start_time
            print(f"DFT Success: {smiles[:20]} ({time_cost:.1f}s)")
            return np.nan_to_num(features), orbital_path
            
        except Exception as e:
            print(f"DFT Error: {smiles} - {str(e)}")
            return np.zeros(len(self.feature_labels)), None
