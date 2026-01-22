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
        cache_file = self.cache_dir / f'{self.data_index[idx]["pdb_id"]}.pt'        
        if cache_file.exists():
            try:
                return torch.load(cache_file, weights_only=False)
            except Exception as e:
                print(f"Cache load failed: {e}")
                cache_file.unlink(missing_ok=True)        
        entry = self.data_index[idx]
        pdb_id = entry['pdb_id']
        ligand_file = self.data_dir / pdb_id / f'{pdb_id}_ligand.sdf'        
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
            y=torch.tensor([deltaG])
        )
def custom_collate(batch):
    batch = [item for item in batch if item is not None and hasattr(item, 'x') and item.x.size(0) > 0]
    if len(batch) == 0:
        return None
    for item in batch:
        if hasattr(item, 'pdb_id'):
            delattr(item, 'pdb_id')
    return Batch.from_data_list(batch)
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