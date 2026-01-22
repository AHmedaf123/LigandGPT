import os
import hashlib
import subprocess
import tempfile
from functools import reduce
from typing import List, Tuple, Optional, Dict, Callable, Any, NamedTuple
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from tdc import Oracle
RDLogger.DisableLog('rdApp.*')
VDW_RADII_MAP = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98
}
IDEAL_TORSION_ANGLES = [-180, -60, 60, 180]
ELECTRONEGATIVITY_MAP = {
    'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98,
    'P': 2.19, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'H': 2.20
}
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOANY
]
class MoleculeData(NamedTuple):
    molecule: Chem.Mol
    molecule_with_hydrogens: Chem.Mol
    has_conformer: bool
    smiles: str
    steric_clash_before_optimization: Optional[float]
    torsion_strain_before_optimization: Optional[float]
class PropertyScores(NamedTuple):
    qed: Optional[float]
    synthetic_accessibility: Optional[float]
    steric_clash: Optional[float]
    torsion_strain: Optional[float]
    internal_energy: Optional[float]
class MoleculeScoreResult(NamedTuple):
    docking_score: Optional[float]
    qed_score: Optional[float]
    synthetic_accessibility_score: Optional[float]
    steric_clash_score: Optional[float]
    torsion_strain_score: Optional[float]
    internal_energy_score: Optional[float]
    bioactivity_score: Optional[float]
def identity(value: Any) -> Any:
    return value
def compose(*functions: Callable) -> Callable:
    return reduce(lambda f, g: lambda x: f(g(x)), functions, identity)
def safe_execute(function: Callable, default_value: Any = None) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            return default_value
    return wrapper
def create_one_hot_encoding(value: Any, possible_values: List[Any]) -> List[int]:
    return [1 if value == v else 0 for v in possible_values]
def generate_hash_identifier(content: str, length: int = 12) -> str:
    return hashlib.md5(content.encode()).hexdigest()[:length]
def normalize_dihedral_angle(angle: float) -> float:
    normalized = angle
    while normalized > 180:
        normalized -= 360
    while normalized < -180:
        normalized += 360
    return normalized
def compute_minimum_angle_deviation(angle: float, reference_angles: List[float]) -> float:
    deviations = [abs(angle - ref) for ref in reference_angles]
    return min(60.0, min(deviations))
def parse_smiles_to_molecule(smiles: str) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(smiles)
def add_hydrogens_to_molecule(molecule: Chem.Mol) -> Chem.Mol:
    return Chem.AddHs(molecule)
def embed_molecule_conformer(molecule: Chem.Mol, random_seed: int = 42) -> bool:
    embedding_parameters = AllChem.ETKDGv3()
    embedding_parameters.randomSeed = random_seed
    return AllChem.EmbedMolecule(molecule, embedding_parameters) == 0
def optimize_molecule_geometry(molecule: Chem.Mol, max_iterations: int = 200) -> None:
    AllChem.UFFOptimizeMolecule(molecule, maxIters=max_iterations)
def extract_atom_positions(molecule: Chem.Mol) -> List[Any]:
    conformer = molecule.GetConformer()
    return [conformer.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())]
def get_atom_neighbor_indices(atom: Chem.Atom) -> set:
    return {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
def calculate_vdw_contact_distance(atomic_number_a: int, atomic_number_b: int) -> float:
    radius_a = VDW_RADII_MAP.get(atomic_number_a, 1.70)
    radius_b = VDW_RADII_MAP.get(atomic_number_b, 1.70)
    return 0.8 * (radius_a + radius_b)
def compute_atom_pair_clash(
    molecule: Chem.Mol,
    positions: List[Any],
    atom_index_a: int,
    atom_index_b: int
) -> float:
    if molecule.GetBondBetweenAtoms(atom_index_a, atom_index_b):
        return 0.0    
    atom_a = molecule.GetAtomWithIdx(atom_index_a)
    atom_b = molecule.GetAtomWithIdx(atom_index_b)    
    neighbors_a = get_atom_neighbor_indices(atom_a)
    neighbors_b = get_atom_neighbor_indices(atom_b)    
    if neighbors_a.intersection(neighbors_b):
        return 0.0    
    distance = positions[atom_index_a].Distance(positions[atom_index_b])
    minimum_allowed_distance = calculate_vdw_contact_distance(
        atom_a.GetAtomicNum(),
        atom_b.GetAtomicNum()
    )   
    return max(0.0, minimum_allowed_distance - distance)
def compute_steric_clash_score(molecule: Chem.Mol) -> Optional[float]:
    try:
        num_atoms = molecule.GetNumAtoms()        
        if num_atoms > 100:
            return 0.0        
        positions = extract_atom_positions(molecule)        
        clash_contributions = [
            compute_atom_pair_clash(molecule, positions, i, j)
            for i in range(num_atoms)
            for j in range(i + 1, num_atoms)
        ]        
        return min(100.0, sum(clash_contributions))
    except Exception:
        return None
def get_rotatable_bond_neighbors(
    molecule: Chem.Mol,
    begin_index: int,
    end_index: int
) -> Tuple[List[int], List[int]]:
    begin_neighbors = [
        n.GetIdx()
        for n in molecule.GetAtomWithIdx(begin_index).GetNeighbors()
        if n.GetIdx() != end_index
    ]
    end_neighbors = [
        n.GetIdx()
        for n in molecule.GetAtomWithIdx(end_index).GetNeighbors()
        if n.GetIdx() != begin_index
    ]
    return begin_neighbors, end_neighbors
def compute_bond_torsion_strain(
    molecule: Chem.Mol,
    conformer: Any,
    bond: Chem.Bond
) -> float:
    if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
        return 0.0    
    begin_index = bond.GetBeginAtomIdx()
    end_index = bond.GetEndAtomIdx()    
    begin_neighbors, end_neighbors = get_rotatable_bond_neighbors(
        molecule, begin_index, end_index
    )    
    if not begin_neighbors or not end_neighbors:
        return 0.0    
    dihedral_angle = rdMolTransforms.GetDihedralDeg(
        conformer,
        begin_neighbors[0],
        begin_index,
        end_index,
        end_neighbors[0]
    )    
    normalized_angle = normalize_dihedral_angle(dihedral_angle)
    deviation = compute_minimum_angle_deviation(normalized_angle, IDEAL_TORSION_ANGLES)    
    return (deviation / 60.0) ** 2
def compute_torsion_strain_score(molecule: Chem.Mol) -> Optional[float]:
    try:
        conformer = molecule.GetConformer()
        strain_contributions = [
            compute_bond_torsion_strain(molecule, conformer, bond)
            for bond in molecule.GetBonds()
        ]
        return sum(strain_contributions)
    except Exception:
        return None
def compute_internal_energy(molecule: Chem.Mol) -> Optional[float]:
    try:
        force_field = UFFGetMoleculeForceField(molecule)
        return force_field.CalcEnergy() if force_field else None
    except Exception:
        return None
def extract_oracle_score(oracle_result: Any) -> float:
    if isinstance(oracle_result, (list, tuple)):
        return float(oracle_result[0])
    return float(oracle_result)
def validate_qed_score(score: float) -> Optional[float]:
    return score if 0 <= score <= 1 else None
def validate_synthetic_accessibility_score(score: float) -> Optional[float]:
    return score if 1.0 <= score <= 10.0 else None
def compute_qed_score(smiles: str, qed_oracle: Oracle) -> Optional[float]:
    try:
        raw_score = qed_oracle(smiles)
        extracted_score = extract_oracle_score(raw_score)
        return validate_qed_score(extracted_score)
    except Exception:
        return None
def compute_synthetic_accessibility_score(smiles: str, sa_oracle: Oracle) -> Optional[float]:
    try:
        raw_score = sa_oracle(smiles)
        extracted_score = extract_oracle_score(raw_score)
        return validate_synthetic_accessibility_score(extracted_score)
    except Exception:
        return None
def build_molecule_data(
    smiles: str,
    molecule: Chem.Mol,
    molecule_with_hydrogens: Chem.Mol,
    has_conformer: bool,
    steric_clash: Optional[float],
    torsion_strain: Optional[float]
) -> MoleculeData:
    return MoleculeData(
        molecule=molecule,
        molecule_with_hydrogens=molecule_with_hydrogens,
        has_conformer=has_conformer,
        smiles=smiles,
        steric_clash_before_optimization=steric_clash,
        torsion_strain_before_optimization=torsion_strain
    )
def prepare_molecule_with_conformer(smiles: str) -> Optional[MoleculeData]:
    molecule = parse_smiles_to_molecule(smiles)
    if molecule is None:
        return None    
    molecule_with_hydrogens = add_hydrogens_to_molecule(molecule)
    conformer_generated = embed_molecule_conformer(molecule_with_hydrogens)    
    if not conformer_generated:
        return build_molecule_data(
            smiles, molecule, molecule_with_hydrogens, False, None, None
        )    
    steric_clash = compute_steric_clash_score(molecule_with_hydrogens)
    torsion_strain = compute_torsion_strain_score(molecule_with_hydrogens)    
    optimize_molecule_geometry(molecule_with_hydrogens)    
    return build_molecule_data(
        smiles, molecule, molecule_with_hydrogens, True, steric_clash, torsion_strain
    )
def create_molecule_data_with_cache(
    smiles: str,
    cache: Dict[str, MoleculeData]
) -> Tuple[Optional[MoleculeData], Dict[str, MoleculeData]]:
    if smiles in cache:
        return cache[smiles], cache    
    molecule_data = prepare_molecule_with_conformer(smiles)    
    if molecule_data is None:
        return None, cache    
    updated_cache = cache.copy()
    if len(updated_cache) < 1000:
        updated_cache[smiles] = molecule_data    
    return molecule_data, updated_cache
def evaluate_property_scores(
    molecule_data: MoleculeData,
    qed_oracle: Oracle,
    sa_oracle: Oracle
) -> PropertyScores:
    qed = compute_qed_score(molecule_data.smiles, qed_oracle)
    synthetic_accessibility = compute_synthetic_accessibility_score(
        molecule_data.smiles, sa_oracle
    )    
    if molecule_data.has_conformer:
        return PropertyScores(
            qed=qed,
            synthetic_accessibility=synthetic_accessibility,
            steric_clash=molecule_data.steric_clash_before_optimization,
            torsion_strain=molecule_data.torsion_strain_before_optimization,
            internal_energy=compute_internal_energy(molecule_data.molecule_with_hydrogens)
        )    
    return PropertyScores(
        qed=qed,
        synthetic_accessibility=synthetic_accessibility,
        steric_clash=None,
        torsion_strain=None,
        internal_energy=None
    )
def determine_receptor_file_path(receptor_file: str) -> Optional[str]:
    if receptor_file.endswith('.pdb'):
        pdbqt_path = receptor_file.replace('.pdb', '.pdbqt')
        return pdbqt_path if os.path.exists(pdbqt_path) else None
    return receptor_file
def generate_ligand_file_paths(temp_directory: str, smiles: str) -> Tuple[str, str, str]:
    molecule_hash = generate_hash_identifier(smiles)
    ligand_pdbqt = os.path.join(temp_directory, f"lig_{molecule_hash}.pdbqt")
    docked_pdbqt = os.path.join(temp_directory, f"dock_{molecule_hash}.pdbqt")
    ligand_sdf = os.path.join(temp_directory, f"lig_{molecule_hash}.sdf")
    return ligand_pdbqt, docked_pdbqt, ligand_sdf
def prepare_ligand_molecule(smiles: str) -> Optional[Chem.Mol]:
    molecule = parse_smiles_to_molecule(smiles)
    if molecule is None:
        return None    
    molecule_with_hydrogens = add_hydrogens_to_molecule(molecule)    
    if AllChem.EmbedMolecule(molecule_with_hydrogens, randomSeed=42, maxAttempts=3) != 0:
        return None    
    AllChem.UFFOptimizeMolecule(molecule_with_hydrogens, maxIters=100)
    return molecule_with_hydrogens
def write_molecule_to_sdf(molecule: Chem.Mol, sdf_path: str) -> bool:
    try:
        with Chem.SDWriter(sdf_path) as writer:
            writer.write(molecule)
        return True
    except Exception:
        return False
def convert_sdf_to_pdbqt(sdf_path: str, pdbqt_path: str) -> bool:
    try:
        subprocess.run(
            ['obabel', sdf_path, '-O', pdbqt_path, '-xh'],
            check=True,
            capture_output=True,
            timeout=5
        )
        return True
    except Exception:
        return False
def build_qvina_command(
    receptor_path: str,
    ligand_path: str,
    output_path: str,
    box_center: List[float],
    box_size: List[int]
) -> List[str]:
    qvina_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docking", "qvina02")
    return [
        qvina_path,
        "--receptor", receptor_path,
        "--ligand", ligand_path,
        "--center_x", str(box_center[0]),
        "--center_y", str(box_center[1]),
        "--center_z", str(box_center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--out", output_path,
        "--exhaustiveness", "4"
    ]
def execute_qvina_docking(command: List[str]) -> bool:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except Exception:
        return False
def parse_docking_score_from_file(file_path: str) -> Optional[float]:
    if not os.path.exists(file_path):
        return None    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'REMARK VINA RESULT:' in line:
                    parts = line.split()
                    score = float(parts[3])
                    if -20 <= score <= 0:
                        return score
    except Exception:
        pass    
    return None
def cleanup_temporary_files(file_paths: List[str]) -> None:
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
def perform_qvina_docking(
    smiles: str,
    receptor_file: str,
    box_center: List[float],
    temp_directory: str,
    box_size: List[int] = None
) -> Optional[float]:
    if box_size is None:
        box_size = [20, 20, 20]    
    receptor_path = determine_receptor_file_path(receptor_file)
    if receptor_path is None:
        return None    
    ligand_pdbqt, docked_pdbqt, ligand_sdf = generate_ligand_file_paths(
        temp_directory, smiles
    )    
    try:
        molecule = prepare_ligand_molecule(smiles)
        if molecule is None:
            return None        
        if not write_molecule_to_sdf(molecule, ligand_sdf):
            return None        
        if not convert_sdf_to_pdbqt(ligand_sdf, ligand_pdbqt):
            return None        
        command = build_qvina_command(
            receptor_path, ligand_pdbqt, docked_pdbqt, box_center, box_size
        )        
        if not execute_qvina_docking(command):
            return None        
        return parse_docking_score_from_file(docked_pdbqt)    
    finally:
        cleanup_temporary_files([ligand_sdf, ligand_pdbqt, docked_pdbqt])
def extract_atom_type_features(symbol: str) -> List[int]:
    one_hot = create_one_hot_encoding(symbol, ATOM_TYPES)
    other_type = 1 if symbol not in ATOM_TYPES else 0
    return one_hot + [other_type]
def extract_degree_features(degree: int) -> List[int]:
    return create_one_hot_encoding(degree, list(range(7)))
def extract_formal_charge_features(charge: int) -> List[int]:
    return create_one_hot_encoding(charge, [-2, -1, 0, 1, 2])
def extract_hybridization_features(hybridization: Any) -> List[int]:
    one_hot = create_one_hot_encoding(hybridization, HYBRIDIZATION_TYPES)
    other_type = 1 if hybridization not in HYBRIDIZATION_TYPES else 0
    return one_hot + [other_type]
def extract_hydrogen_count_features(num_hydrogens: int) -> List[int]:
    return create_one_hot_encoding(num_hydrogens, list(range(5)))
def extract_ring_size_features(atom: Chem.Atom) -> List[int]:
    ring_sizes = [3, 4, 5, 6, 7, 8]
    features = [0] * 6    
    if atom.IsInRing():
        for i, size in enumerate(ring_sizes):
            if atom.IsInRingSize(size):
                features[i] = 1
                break    
    other_ring = 1 if atom.IsInRing() and sum(features) == 0 else 0
    return features + [other_ring]
def extract_valence_features(valence: int) -> List[int]:
    return create_one_hot_encoding(valence, list(range(7)))
def extract_bond_count_features(num_bonds: int) -> List[int]:
    return create_one_hot_encoding(num_bonds, list(range(7)))
def extract_radical_features(num_radicals: int) -> List[int]:
    return create_one_hot_encoding(num_radicals, list(range(4)))
def extract_implicit_hydrogen_features(num_implicit_h: int) -> List[int]:
    return create_one_hot_encoding(num_implicit_h, list(range(7)))
def extract_chirality_features(atom: Chem.Atom) -> List[int]:
    has_chirality = 1 if atom.HasProp('_ChiralityPossible') else 0
    chiral_tag = atom.GetChiralTag()
    is_cw = 1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0
    is_ccw = 1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0
    return [has_chirality, is_cw, is_ccw]
def build_atom_feature_vector(atom: Chem.Atom) -> List[float]:
    symbol = atom.GetSymbol()    
    features = []
    features.extend(extract_atom_type_features(symbol))
    features.extend(extract_degree_features(atom.GetDegree()))
    features.extend(extract_formal_charge_features(atom.GetFormalCharge()))
    features.extend(extract_hybridization_features(atom.GetHybridization()))
    features.extend(extract_hydrogen_count_features(atom.GetTotalNumHs()))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    features.extend(extract_chirality_features(atom))
    features.extend(extract_valence_features(atom.GetTotalValence()))
    features.append(atom.GetMass() / 100.0)
    features.extend(extract_ring_size_features(atom))
    features.extend(extract_bond_count_features(len(atom.GetBonds())))
    features.extend(extract_radical_features(atom.GetNumRadicalElectrons()))
    features.append(ELECTRONEGATIVITY_MAP.get(symbol, 2.5) / 4.0)
    features.extend(extract_implicit_hydrogen_features(atom.GetNumImplicitHs()))
    features.append(atom.GetAtomicNum() / 100.0)
    features.append(1 if symbol not in ['C', 'H'] else 0)    
    return features
def extract_bond_type_features(bond_type: Any) -> List[int]:
    return create_one_hot_encoding(bond_type, BOND_TYPES)
def extract_stereo_features(stereo: Any) -> List[int]:
    return create_one_hot_encoding(stereo, STEREO_TYPES)
def extract_bond_direction_features(bond_direction: Any) -> List[int]:
    is_up = 1 if bond_direction == Chem.rdchem.BondDir.ENDUPRIGHT else 0
    is_down = 1 if bond_direction == Chem.rdchem.BondDir.ENDDOWNRIGHT else 0
    return [is_up, is_down]
def build_bond_feature_vector(bond: Chem.Bond) -> List[float]:
    features = []
    features.extend(extract_bond_type_features(bond.GetBondType()))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    features.extend(extract_stereo_features(bond.GetStereo()))
    features.extend(extract_bond_direction_features(bond.GetBondDir()))
    return features
def convert_smiles_to_graph(smiles: str) -> Optional[Data]:
    molecule = parse_smiles_to_molecule(smiles)
    if molecule is None:
        return None    
    molecule = add_hydrogens_to_molecule(molecule)    
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None    
    node_features = [build_atom_feature_vector(atom) for atom in molecule.GetAtoms()]    
    edge_indices = []
    edge_features = []    
    for bond in molecule.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_features = build_bond_feature_vector(bond)        
        edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_features.extend([bond_features, bond_features])    
    if not edge_indices:
        return None    
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_features, dtype=torch.float)
    )
def clip_bioactivity_prediction(prediction: float) -> float:
    return float(np.clip(prediction, -20.0, 0.0))
def predict_bioactivity_for_batch(
    smiles_list: List[str],
    gnn_model: Any
) -> List[Optional[float]]:
    if gnn_model is None:
        return [None] * len(smiles_list)    
    try:
        graphs_with_indices = [
            (i, convert_smiles_to_graph(smiles))
            for i, smiles in enumerate(smiles_list)
        ]        
        valid_entries = [
            (idx, graph)
            for idx, graph in graphs_with_indices
            if graph is not None
        ]        
        if not valid_entries:
            return [None] * len(smiles_list)        
        valid_indices, valid_graphs = zip(*valid_entries)
        batch_graph = Batch.from_data_list(list(valid_graphs))        
        with torch.no_grad():
            predictions = gnn_model(batch_graph)
            if predictions.dim() == 2:
                predictions = predictions.squeeze(-1)
            predictions_array = predictions.cpu().numpy()        
        results = [None] * len(smiles_list)
        for original_index, prediction in zip(valid_indices, predictions_array):
            results[original_index] = clip_bioactivity_prediction(prediction)        
        return results    
    except Exception:
        return [None] * len(smiles_list)
def generate_cache_key(smiles: str, receptor_file: str) -> str:
    receptor_hash = generate_hash_identifier(receptor_file)
    return f"{smiles}_{receptor_hash}"
def build_molecule_score_result(
    docking: Optional[float],
    property_scores: PropertyScores,
    bioactivity: Optional[float]
) -> MoleculeScoreResult:
    return MoleculeScoreResult(
        docking_score=docking,
        qed_score=property_scores.qed,
        synthetic_accessibility_score=property_scores.synthetic_accessibility,
        steric_clash_score=property_scores.steric_clash,
        torsion_strain_score=property_scores.torsion_strain,
        internal_energy_score=property_scores.internal_energy,
        bioactivity_score=bioactivity
    )
def score_single_molecule(
    smiles: str,
    receptor_file: str,
    box_center: List[float],
    temp_directory: str,
    qed_oracle: Oracle,
    sa_oracle: Oracle,
    gnn_model: Any,
    molecule_cache: Dict[str, MoleculeData]
) -> Tuple[MoleculeScoreResult, Dict[str, MoleculeData]]:
    molecule_data, updated_cache = create_molecule_data_with_cache(smiles, molecule_cache)    
    if molecule_data is None:
        empty_properties = PropertyScores(None, None, None, None, None)
        return build_molecule_score_result(None, empty_properties, None), updated_cache    
    property_scores = evaluate_property_scores(molecule_data, qed_oracle, sa_oracle)
    docking_score = perform_qvina_docking(smiles, receptor_file, box_center, temp_directory)
    bioactivity_predictions = predict_bioactivity_for_batch([smiles], gnn_model)
    bioactivity_score = bioactivity_predictions[0]    
    result = build_molecule_score_result(docking_score, property_scores, bioactivity_score)
    return result, updated_cache
def convert_score_result_to_list(result: MoleculeScoreResult) -> List[Optional[float]]:
    return [
        result.docking_score,
        result.qed_score,
        result.synthetic_accessibility_score,
        result.steric_clash_score,
        result.torsion_strain_score,
        result.internal_energy_score,
        result.bioactivity_score
    ]
def calculate_scores_for_batch(
    smiles_batch: List[str],
    receptor_file: str,
    box_center: List[float],
    gnn_model: Any,
    qed_oracle: Oracle,
    sa_oracle: Oracle,
    score_cache: Dict[str, List[Optional[float]]],
    molecule_cache: Dict[str, MoleculeData],
    temp_directory: str
) -> List[List[Optional[float]]]:
    accumulated_scores = []
    current_molecule_cache = molecule_cache.copy()
    current_score_cache = score_cache.copy()        
    for smiles in smiles_batch:
        cache_key = generate_cache_key(smiles, receptor_file)            
        if cache_key in current_score_cache:
            accumulated_scores.append(current_score_cache[cache_key])
            continue            
        score_result, current_molecule_cache = score_single_molecule(
            smiles,
            receptor_file,
            box_center,
            temp_directory,
            qed_oracle,
            sa_oracle,
            gnn_model,
            current_molecule_cache
        )        
        score_list = convert_score_result_to_list(score_result)
        current_score_cache[cache_key] = score_list
        accumulated_scores.append(score_list)        
    return accumulated_scores
def create_qed_oracle() -> Oracle:
    return Oracle(name='QED')
def create_synthetic_accessibility_oracle() -> Oracle:
    return Oracle(name='SA')
def initialize_scoring_oracles() -> Dict[str, Oracle]:
    return {
        'qed': create_qed_oracle(),
        'sa': create_synthetic_accessibility_oracle()
    }
def strip_module_prefix_from_state_dict(state_dict: Dict) -> Dict:
    return {key.replace('module.', ''): value for key, value in state_dict.items()}
def load_gnn_model_from_checkpoint(model_path: str) -> Optional[Any]:
    if not os.path.exists(model_path):
        print(f"WARNING: Bioactivity GNN checkpoint not found at {model_path}")
        print("Bioactivity scores will not be computed (returning None for all molecules)")
        return None    
    try:
        import sys
        gnn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GNN')
        if gnn_dir not in sys.path:
            sys.path.append(gnn_dir)
        from GNNmodel import AdvancedGNN        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = strip_module_prefix_from_state_dict(checkpoint['model_state_dict'])        
        model = AdvancedGNN(
            node_input_dim=75,
            edge_input_dim=12,
            hidden_dim=256,
            num_gin_layers=5,
            num_gat_layers=2,
            gat_heads=8,
            dropout=0.3,
            num_tasks=1
        )        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"Successfully loaded bioactivity GNN model from {model_path}")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load bioactivity GNN model: {e}")
        print("Bioactivity scores will not be computed (returning None for all molecules)")
        return None
def create_temporary_docking_directory() -> str:
    return tempfile.mkdtemp(prefix='docking_')
def calculate_tanimoto_diversity(smiles_list: List[str]) -> float:
    from tdc import Evaluator
    diversity_evaluator = Evaluator(name='Diversity')
    return diversity_evaluator(smiles_list)
_SCORING_ORACLES = None
_GNN_MODEL = None
_TEMP_DIR = None
def initialize_scoring_system(gnn_model_path: str = None):
    global _SCORING_ORACLES, _GNN_MODEL, _TEMP_DIR    
    if gnn_model_path is None:
        gnn_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'GNN', 'outputs', 'pdbbind_finetune', 'best_model.pt'
        )    
    if _SCORING_ORACLES is None:
        _SCORING_ORACLES = initialize_scoring_oracles()    
    if _GNN_MODEL is None:
        _GNN_MODEL = load_gnn_model_from_checkpoint(gnn_model_path)    
    if _TEMP_DIR is None:
        _TEMP_DIR = create_temporary_docking_directory()    
    return _SCORING_ORACLES, _GNN_MODEL, _TEMP_DIR
def calculate_scores(
    smiles_list: List[str],
    receptor_file: str,
    box_center: List[float]
) -> List[List[Optional[float]]]:
    oracles, gnn_model, temp_dir = initialize_scoring_system()    
    return calculate_scores_for_batch(
        smiles_batch=smiles_list,
        receptor_file=receptor_file,
        box_center=box_center,
        gnn_model=gnn_model,
        qed_oracle=oracles['qed'],
        sa_oracle=oracles['sa'],
        score_cache={},
        molecule_cache={},
        temp_directory=temp_dir
    )
def cleanup_scoring_system():
    global _TEMP_DIR
    if _TEMP_DIR and os.path.exists(_TEMP_DIR):
        import shutil
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    _TEMP_DIR = None