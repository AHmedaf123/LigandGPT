import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import sqlite3
import pickle
from pathlib import Path
from tqdm import tqdm

# Protein pocket atom types (common elements in protein pockets)
PROT_ATOM_TYPES = ['C', 'N', 'O', 'S', 'H', 'FE', 'ZN', 'MG', 'CA', 'MN']
# Standard amino acid residue types
RESIDUE_TYPES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}


class ProteinPocketFeaturizer:
    """Featurizes protein pocket atoms from a PDB file.
    
    Produces a 40-dimensional feature vector per pocket atom:
      - Element type one-hot (10 types + 1 other = 11)
      - Residue type one-hot (20 standard amino acids + 1 other = 21)
      - Is backbone atom (1)
      - Is sidechain atom (1)
      - Normalized B-factor (1)
      - 3D coordinates x, y, z (3, normalized by /10)
      - Normalized occupancy (1)
      - Charge proxy from element (1)
    Total: 11 + 21 + 2 + 1 + 3 + 1 + 1 = 40
    """
    CHARGE_MAP = {
        'C': 0.0, 'N': -0.1, 'O': -0.2, 'S': -0.05,
        'H': 0.1, 'FE': 0.3, 'ZN': 0.2, 'MG': 0.2,
        'CA': 0.2, 'MN': 0.2
    }
    
    @staticmethod
    def parse_pocket_pdb(pdb_path):
        """Parse a pocket PDB file and extract atom records.
        
        Returns:
            list of dicts with keys: element, resname, atom_name, coords, bfactor, occupancy
        """
        atoms = []
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        try:
                            atom_name = line[12:16].strip()
                            resname = line[17:20].strip()
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            occupancy = float(line[54:60]) if len(line) > 60 else 1.0
                            bfactor = float(line[60:66]) if len(line) > 66 else 0.0
                            element = line[76:78].strip().upper() if len(line) > 76 else atom_name[0].upper()
                            atoms.append({
                                'element': element,
                                'resname': resname,
                                'atom_name': atom_name,
                                'coords': [x, y, z],
                                'bfactor': bfactor,
                                'occupancy': occupancy
                            })
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        return atoms
    
    @staticmethod
    def featurize_pocket(pdb_path, max_pocket_atoms=300):
        """Convert pocket PDB to feature tensor.
        
        Args:
            pdb_path: path to the pocket PDB file
            max_pocket_atoms: maximum number of pocket atoms to keep
            
        Returns:
            torch.Tensor of shape [num_atoms, 40] or None if parsing fails
        """
        atoms = ProteinPocketFeaturizer.parse_pocket_pdb(pdb_path)
        if not atoms:
            return None
        
        # Filter out hydrogen atoms for efficiency
        atoms = [a for a in atoms if a['element'] != 'H']
        
        # Truncate if too many atoms (keep closest to center)
        if len(atoms) > max_pocket_atoms:
            atoms = atoms[:max_pocket_atoms]
        
        if not atoms:
            return None
        
        features_list = []
        for atom in atoms:
            feat = []
            
            # Element type one-hot (10 + 1 other = 11)
            elem = atom['element']
            elem_onehot = [1 if elem == t else 0 for t in PROT_ATOM_TYPES]
            elem_onehot.append(1 if elem not in PROT_ATOM_TYPES else 0)
            feat.extend(elem_onehot)
            
            # Residue type one-hot (20 + 1 other = 21)
            resname = atom['resname']
            res_onehot = [1 if resname == r else 0 for r in RESIDUE_TYPES]
            res_onehot.append(1 if resname not in RESIDUE_TYPES else 0)
            feat.extend(res_onehot)
            
            # Is backbone / sidechain (2)
            is_backbone = 1 if atom['atom_name'] in BACKBONE_ATOMS else 0
            feat.append(is_backbone)
            feat.append(1 - is_backbone)  # is_sidechain
            
            # Normalized B-factor (1)
            feat.append(min(atom['bfactor'] / 100.0, 1.0))
            
            # 3D coordinates normalized (3)
            feat.extend([c / 10.0 for c in atom['coords']])
            
            # Occupancy (1)
            feat.append(atom['occupancy'])
            
            # Charge proxy (1)
            feat.append(ProteinPocketFeaturizer.CHARGE_MAP.get(elem, 0.0))
            
            assert len(feat) == 40, f"Expected 40 protein atom features, got {len(feat)}"
            features_list.append(feat)
        
        return torch.tensor(features_list, dtype=torch.float32)
    
    @staticmethod
    def featurize_from_atom_names(atom_names, coords=None):
        """Create protein pocket features from atom name list and optional coordinates.
        
        This is used during RL generation where protein data comes from the 
        tokenizer's atom_name list rather than a PDB file.
        
        Args:
            atom_names: list of atom name strings (e.g., ['N', 'CA', 'C', 'O', ...])
            coords: optional numpy array of shape [num_atoms, 3]
            
        Returns:
            torch.Tensor of shape [num_atoms, 40] or None
        """
        if not atom_names:
            return None
        
        # Filter out hydrogens
        if coords is not None:
            non_h = [(name, coords[i]) for i, name in enumerate(atom_names) if not name.startswith('H')]
        else:
            non_h = [(name, [0.0, 0.0, 0.0]) for name in atom_names if not name.startswith('H')]
        
        if not non_h:
            return None
        
        features_list = []
        for atom_name, coord in non_h:
            feat = []
            
            # Element type: extract from atom name (first char usually)
            elem = atom_name[0].upper() if atom_name else 'C'
            if atom_name == 'CA':  # alpha carbon, not calcium
                elem = 'C'
            
            # Element type one-hot (11)
            elem_onehot = [1 if elem == t else 0 for t in PROT_ATOM_TYPES]
            elem_onehot.append(1 if elem not in PROT_ATOM_TYPES else 0)
            feat.extend(elem_onehot)
            
            # Residue type: unknown from atom names alone (21 zeros + 1 other)
            feat.extend([0] * 20 + [1])
            
            # Is backbone / sidechain (2)
            is_backbone = 1 if atom_name in BACKBONE_ATOMS else 0
            feat.append(is_backbone)
            feat.append(1 - is_backbone)
            
            # B-factor: unknown (1)
            feat.append(0.0)
            
            # 3D coordinates (3)
            if isinstance(coord, (list, tuple)):
                feat.extend([c / 10.0 for c in coord])
            else:
                feat.extend([coord[0] / 10.0, coord[1] / 10.0, coord[2] / 10.0])
            
            # Occupancy: unknown (1)
            feat.append(1.0)
            
            # Charge proxy (1)
            feat.append(ProteinPocketFeaturizer.CHARGE_MAP.get(elem, 0.0))
            
            features_list.append(feat)
        
        return torch.tensor(features_list, dtype=torch.float32)
class MoleculeFeaturizer:
    @staticmethod
    def atom_features(atom):
        features = []
        atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
        features.extend([1 if atom.GetSymbol() == t else 0 for t in atom_types])
        features.append(1 if atom.GetSymbol() not in atom_types else 0)
        features.extend([1 if atom.GetDegree() == i else 0 for i in range(7)])
        formal_charges = [-2, -1, 0, 1, 2]
        features.extend([1 if atom.GetFormalCharge() == c else 0 for c in formal_charges])
        hyb_types = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3,
                     Chem.rdchem.HybridizationType.SP3D,
                     Chem.rdchem.HybridizationType.SP3D2]
        features.extend([1 if atom.GetHybridization() == t else 0 for t in hyb_types])
        features.append(1 if atom.GetHybridization() not in hyb_types else 0)
        features.extend([1 if atom.GetTotalNumHs() == i else 0 for i in range(5)])
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(1 if atom.IsInRing() else 0)
        features.append(1 if atom.HasProp('_ChiralityPossible') else 0)
        try:
            chiral_type = atom.GetChiralTag()
            features.append(1 if chiral_type == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0)
            features.append(1 if chiral_type == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0)
        except:
            features.extend([0, 0])
        features.extend([1 if atom.GetTotalValence() == i else 0 for i in range(7)])
        features.append(atom.GetMass() / 100.0)
        ring_sizes = [3, 4, 5, 6, 7, 8]
        in_ring_size = [0] * 6
        if atom.IsInRing():
            for i, size in enumerate(ring_sizes):
                if atom.IsInRingSize(size):
                    in_ring_size[i] = 1
                    break
        features.extend(in_ring_size)
        features.append(1 if atom.IsInRing() and sum(in_ring_size) == 0 else 0)
        num_bonds = len(atom.GetBonds())
        features.extend([1 if num_bonds == i else 0 for i in range(7)])
        features.extend([1 if atom.GetNumRadicalElectrons() == i else 0 for i in range(4)])
        electroneg = {'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98,
                     'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'H': 2.20}
        features.append(electroneg.get(atom.GetSymbol(), 2.5) / 4.0)
        impl_h = atom.GetNumImplicitHs()
        features.extend([1 if impl_h == i else 0 for i in range(7)])
        features.append(atom.GetAtomicNum() / 100.0)
        features.append(1 if atom.GetSymbol() not in ['C', 'H'] else 0)        
        assert len(features) == 75, f"Expected 75 features, got {len(features)}"
        return features    
    @staticmethod
    def bond_features(bond):
        if bond is None:
            return [0] * 12
        features = []
        bond_types = [Chem.rdchem.BondType.SINGLE,
                      Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE,
                      Chem.rdchem.BondType.AROMATIC]
        features.extend([1 if bond.GetBondType() == t else 0 for t in bond_types])
        features.append(1 if bond.GetIsConjugated() else 0)
        features.append(1 if bond.IsInRing() else 0)
        stereo_types = [Chem.rdchem.BondStereo.STEREONONE,
                        Chem.rdchem.BondStereo.STEREOZ,
                        Chem.rdchem.BondStereo.STEREOE,
                        Chem.rdchem.BondStereo.STEREOANY]
        features.extend([1 if bond.GetStereo() == t else 0 for t in stereo_types])
        features.append(1 if bond.GetBondDir() == Chem.rdchem.BondDir.ENDUPRIGHT else 0)
        features.append(1 if bond.GetBondDir() == Chem.rdchem.BondDir.ENDDOWNRIGHT else 0)        
        return features    
    @staticmethod
    def mol_to_graph(mol):
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(MoleculeFeaturizer.atom_features(atom))        
        x = torch.tensor(atom_features, dtype=torch.float32)
        edge_indices = []
        edge_features = []        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()            
            edge_indices.extend([[i, j], [j, i]])            
            bond_feat = MoleculeFeaturizer.bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
class ChEMBLDataset(Dataset):
    CACHE_VERSION = 'v5_deltaG_75feat_fixed'    
    def __init__(self, db_path, cache_dir='./cache/chembl', max_atoms=150, 
                 chunk_size=10000, preload=False):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir) / self.CACHE_VERSION
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_atoms = max_atoms
        self.chunk_size = chunk_size
        self.index_file = self.cache_dir / 'index.pkl'
        if self.index_file.exists():
            with open(self.index_file, 'rb') as f:
                self.data_index = pickle.load(f)
            print(f"Loaded index with {len(self.data_index)} entries")
        else:
            print("Building dataset index from ChEMBL database...")
            self.data_index = self._build_index()
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.data_index, f)        
        self.featurizer = MoleculeFeaturizer()        
    def _build_index(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        query = """
        SELECT DISTINCT 
            cs.canonical_smiles,
            act.standard_value,
            act.standard_units,
            act.standard_type
        FROM activities act
        JOIN compound_structures cs ON act.molregno = cs.molregno
        WHERE act.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
        AND act.standard_value IS NOT NULL
        AND act.standard_units = 'nM'
        AND cs.canonical_smiles IS NOT NULL
        AND CAST(act.standard_value AS REAL) > 0
        """        
        data_index = []
        chunk = []        
        try:
            for row in tqdm(cursor.execute(query), desc="Indexing ChEMBL"):
                smiles, value, units, act_type = row
                try:
                    value = float(value)
                    if value > 0:
                        deltaG = 1.36 * (np.log10(value) - 9)
                        if -20.0 <= deltaG <= 0.0:  
                            chunk.append({'smiles': smiles, 'deltaG': deltaG, 'type': act_type})
                            
                            if len(chunk) >= self.chunk_size:
                                data_index.extend(chunk)
                                chunk = []
                except (ValueError, TypeError):
                    continue            
            if chunk:
                data_index.extend(chunk)
        finally:
            conn.close()        
        print(f"Indexed {len(data_index)} molecules")
        return data_index    
    def __len__(self):
        return len(self.data_index)    
    def __getitem__(self, idx):
        cache_file = self.cache_dir / f'mol_{idx}.pt'
        if cache_file.exists():
            try:
                return torch.load(cache_file, weights_only=False)
            except Exception as e:
                print(f"Cache load failed for {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)
        entry = self.data_index[idx]
        mol = Chem.MolFromSmiles(entry['smiles'])
        if mol is None or mol.GetNumAtoms() > self.max_atoms:
            return Data(
                x=torch.zeros((1, 75)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 12)),
                y=torch.tensor([-10.0])  # Neutral ΔG value
            )        
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)
            graph = self.featurizer.mol_to_graph(mol)            
            if graph is None or graph.x.size(0) == 0:
                return Data(
                    x=torch.zeros((1, 75)),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 12)),
                    y=torch.tensor([-10.0])
                )            
            graph.y = torch.tensor([entry['deltaG']], dtype=torch.float32)
            if torch.isnan(graph.y).any() or torch.isinf(graph.y).any():
                print(f"Invalid target value for molecule {idx}: {entry['deltaG']}")
                return Data(
                    x=torch.zeros((1, 75)),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 12)),
                    y=torch.tensor([-10.0])
                )
            torch.save(graph, cache_file)            
            return graph
        except Exception as e:
            print(f"Failed to process molecule {idx}: {e}")
            return Data(
                x=torch.zeros((1, 75)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 12)),
                y=torch.tensor([-10.0])
            )
class PDBBindDataset(Dataset):
    def __init__(self, data_dir, index_file, cache_dir='./cache/pdbbind', 
                 max_atoms=150):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_atoms = max_atoms
        self.data_index = self._load_index(index_file)
        self.featurizer = MoleculeFeaturizer()        
    def _load_index(self, index_file):
        data_index = []        
        with open(index_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue                
                parts = line.strip().split()
                if len(parts) >= 4:
                    pdb_id = parts[0]
                    resolution = parts[1]
                    year = parts[2]                    
                    try:
                        pKd = float(parts[3])
                        deltaG = -1.364 * pKd
                        if -20.0 <= deltaG <= 0.0:
                            data_index.append({
                                'pdb_id': pdb_id,
                                'deltaG': deltaG,
                                'pKd': pKd,
                                'resolution': resolution
                            })
                    except (ValueError, IndexError):
                        continue
        print(f"Loaded {len(data_index)} PDBBind entries with ΔG values")
        return data_index    
    def __len__(self):
        return len(self.data_index)    
    def __getitem__(self, idx):
        # Use versioned cache to avoid stale data from old format
        cache_file = self.cache_dir / f'{self.data_index[idx]["pdb_id"]}_v2_protlig.pt'
        if cache_file.exists():
            try:
                return torch.load(cache_file, weights_only=False)
            except Exception as e:
                print(f"Cache load failed: {e}")
                cache_file.unlink(missing_ok=True)        
        entry = self.data_index[idx]
        pdb_id = entry['pdb_id']
        ligand_file = self.data_dir / pdb_id / f'{pdb_id}_ligand.sdf'
        # Look for pocket PDB file (standard PDBBind naming conventions)
        pocket_file = self.data_dir / pdb_id / f'{pdb_id}_pocket.pdb'
        if not pocket_file.exists():
            pocket_file = self.data_dir / pdb_id / f'{pdb_id}_protein.pdb'
        try:
            if not ligand_file.exists():
                return self._dummy_graph(entry['deltaG'])            
            suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
            mol = None
            for m in suppl:
                if m is not None:
                    mol = m
                    break            
            if mol is None or mol.GetNumAtoms() > self.max_atoms:
                return self._dummy_graph(entry['deltaG'])            
            graph = self.featurizer.mol_to_graph(mol)
            if graph is None or graph.x.size(0) == 0:
                return self._dummy_graph(entry['deltaG'])            
            graph.y = torch.tensor([entry['deltaG']], dtype=torch.float32)
            
            # [B3] Add protein pocket features: y_bio = f(P, L)
            if pocket_file.exists():
                prot_featurizer = ProteinPocketFeaturizer()
                prot_x = prot_featurizer.featurize_pocket(str(pocket_file))
                if prot_x is not None and prot_x.size(0) > 0:
                    graph.prot_x = prot_x
                else:
                    graph.prot_x = torch.zeros((1, 40), dtype=torch.float32)
            else:
                graph.prot_x = torch.zeros((1, 40), dtype=torch.float32)
            
            torch.save(graph, cache_file)
            return graph            
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            return self._dummy_graph(entry['deltaG'])    
    def _dummy_graph(self, deltaG):
        return Data(
            x=torch.zeros((1, 75)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 12)),
            y=torch.tensor([deltaG]),
            prot_x=torch.zeros((1, 40), dtype=torch.float32)
        )
def custom_collate(batch):
    """Custom collation that handles both ligand graphs and protein pocket features.
    
    For protein features (prot_x), we manually batch them and create prot_batch 
    indices since PyG's Batch.from_data_list doesn't automatically handle 
    non-standard tensor attributes with variable sizes.
    """
    batch = [item for item in batch if item is not None and hasattr(item, 'x') and item.x.size(0) > 0]
    if len(batch) == 0:
        return None
    for item in batch:
        if hasattr(item, 'pdb_id'):
            delattr(item, 'pdb_id')
    
    # Extract protein features before PyG batching
    prot_x_list = []
    prot_batch_list = []
    for i, item in enumerate(batch):
        if hasattr(item, 'prot_x') and item.prot_x is not None:
            prot_x_list.append(item.prot_x)
            prot_batch_list.append(torch.full((item.prot_x.size(0),), i, dtype=torch.long))
            # Remove prot_x before PyG batching to avoid dimension mismatch issues
            delattr(item, 'prot_x')
    
    # Standard PyG batching for ligand graphs
    batched = Batch.from_data_list(batch)
    
    # Re-attach batched protein features
    if prot_x_list:
        batched.prot_x = torch.cat(prot_x_list, dim=0)
        batched.prot_batch = torch.cat(prot_batch_list, dim=0)
    else:
        batched.prot_x = torch.zeros((len(batch), 40), dtype=torch.float32)
        batched.prot_batch = torch.arange(len(batch), dtype=torch.long)
    
    return batched
def create_dataloaders(dataset, batch_size=32, train_split=0.8, val_split=0.1,
                       num_workers=4, shuffle=True):    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))    
    if shuffle:
        np.random.shuffle(indices)    
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        exclude_keys=['pdb_id'],
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        exclude_keys=['pdb_id'],
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )    
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        exclude_keys=['pdb_id'],
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )    
    return train_loader, val_loader, test_loader