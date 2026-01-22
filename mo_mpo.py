import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, replace
from collections import deque, OrderedDict
import logging
try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_MURCKO = True
except ImportError:
    HAS_MURCKO = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
@dataclass(frozen=True)
class Objective:
    name: str
    vmin: float
    vmax: float
    weight: float = 1.0
    minimize: bool = False
    dynamic_bounds: bool = False
    bounds_momentum: float = 0.25
    observation_frequency: float = 1.0
@dataclass(frozen=True)
class ObjectiveNormalizationParams:
    vmin: float
    vmax: float
    minimize: bool
@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 32
    archive_size: int = 100
    pareto_bonus_base: float = 0.2
    pareto_decay_rate: float = 0.7
    novelty_fp_weight_initial: float = 0.05
    novelty_scaffold_bonus_initial: float = 0.03
    novelty_decay_iterations: int = 5000
    soft_saturation_scale: float = 2.0
    excellence_threshold: float = 0.2
    excellence_max_bonus: float = 0.1
    penalty_syntax: float = -3.0
    penalty_chemistry: float = -2.0
    penalty_conformer: float = -1.0
    penalty_unknown: float = -2.0
    advantage_clip_min: float = -3.0
    advantage_clip_max: float = 3.0
    scaffold_cache_size: int = 2000
    fingerprint_cache_size: int = 500
@dataclass(frozen=True)
class RunningStatistics:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = np.inf
    max_value: float = -np.inf
    values: tuple = ()
    window_size: int = 1000
def create_running_statistics(window_size: int = 1000) -> RunningStatistics:
    return RunningStatistics(window_size=window_size)
def update_running_statistics(stats: RunningStatistics, value: Optional[float]) -> RunningStatistics:
    if value is None:
        return stats    
    new_count = stats.count + 1
    delta = value - stats.mean
    new_mean = stats.mean + delta / new_count
    delta2 = value - new_mean
    new_m2 = stats.m2 + delta * delta2
    new_min = min(stats.min_value, value)
    new_max = max(stats.max_value, value)    
    new_values = list(stats.values) if stats.values else []
    new_values.append(value)
    if len(new_values) > stats.window_size:
        new_values = new_values[-stats.window_size:]    
    return RunningStatistics(
        count=new_count,
        mean=new_mean,
        m2=new_m2,
        min_value=new_min,
        max_value=new_max,
        values=tuple(new_values),
        window_size=stats.window_size
    )
def compute_statistics_std(stats: RunningStatistics) -> float:
    if stats.count < 2:
        return 0.0
    return float(np.sqrt(stats.m2 / (stats.count - 1)))
def compute_statistics_percentile(stats: RunningStatistics, quantile: float) -> float:
    if not stats.values:
        return 0.0
    return float(np.percentile(stats.values, quantile))
def has_sufficient_data(stats: RunningStatistics, min_samples: int = 100) -> bool:
    return stats.count >= min_samples
def reset_running_statistics(stats: RunningStatistics) -> RunningStatistics:
    return create_running_statistics(stats.window_size)
def normalize_objective_value(value: Optional[float], params: ObjectiveNormalizationParams) -> Optional[float]:
    if value is None:
        return None    
    value_range = params.vmax - params.vmin
    if abs(value_range) < 1e-10:
        return 0.5    
    if params.minimize:
        normalized = (params.vmax - value) / value_range
    else:
        normalized = (value - params.vmin) / value_range    
    return float(np.clip(normalized, 0.0, 1.0))
def compute_updated_bounds(
    current_vmin: float,
    current_vmax: float,
    percentile_5: float,
    percentile_95: float,
    minimize: bool,
    momentum: float,
    observation_frequency: float
) -> Tuple[float, float]:
    margin = 0.2
    value_range = max(percentile_95 - percentile_5, 1e-6)    
    if minimize:
        new_vmin = max(0.0, percentile_5 - margin * value_range)
        new_vmax = percentile_95 + margin * value_range
    else:
        new_vmin = percentile_5 - margin * value_range
        new_vmax = percentile_95 + margin * value_range    
    effective_momentum = momentum * observation_frequency
    updated_vmin = (1 - effective_momentum) * current_vmin + effective_momentum * new_vmin
    updated_vmax = (1 - effective_momentum) * current_vmax + effective_momentum * new_vmax    
    return updated_vmin, updated_vmax
def compute_smooth_excellence_bonus(
    distance: float,
    threshold: float,
    max_bonus: float
) -> float:
    if distance >= threshold:
        return 0.0    
    x = (threshold - distance) / threshold
    sigmoid_value = 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))
    return max_bonus * sigmoid_value
def compute_batch_tanimoto_minimum_similarity(
    query_fingerprint,
    reference_fingerprints: List,
    max_references: int = 100
) -> float:
    from rdkit.DataStructs import BulkTanimotoSimilarity    
    if not reference_fingerprints:
        return 0.0    
    references = (reference_fingerprints[-max_references:] 
                  if len(reference_fingerprints) > max_references 
                  else reference_fingerprints)    
    similarities = BulkTanimotoSimilarity(query_fingerprint, references)
    return min(similarities) if similarities else 0.0
def create_objectives() -> List[Objective]:
    return [
        Objective('qed', vmin=0.0, vmax=1.0, minimize=False, observation_frequency=1.0),
        Objective('sa', vmin=1.0, vmax=10.0, minimize=True, observation_frequency=1.0),
        Objective('docking', vmin=-12.0, vmax=0.0, minimize=True, observation_frequency=0.9),
        Objective('energy', vmin=0.0, vmax=200.0, minimize=True, observation_frequency=0.7),
        Objective('bioactivity', vmin=-12.0, vmax=0.0, minimize=True, observation_frequency=0.9),
        Objective('steric', vmin=0.0, vmax=20.0, minimize=True, observation_frequency=0.6),
        Objective('torsion', vmin=0.0, vmax=5.0, minimize=True, observation_frequency=0.6),
    ]
def map_energy_to_reward(energy: Optional[float]) -> float:
    if energy is None:
        return 0.0
    return map_piecewise_to_reward(energy, good_threshold=0.0, bad_threshold=200.0, minimize=True)
def map_sa_to_reward(sa_score: Optional[float]) -> float:
    if sa_score is None:
        return 0.0
    if sa_score <= 2.5:
        return 1.0
    if sa_score >= 10.0:
        return 0.0
    return float(max(0.0, 1.0 - (sa_score - 2.5) / (10.0 - 2.5)))
def map_piecewise_to_reward(
    value: Optional[float],
    good_threshold: float,
    bad_threshold: float,
    minimize: bool = True
) -> float:
    if value is None:
        return 0.0    
    if minimize:
        if value <= good_threshold:
            return 1.0
        if value >= bad_threshold:
            return 0.0
        return float(1.0 - (value - good_threshold) / (bad_threshold - good_threshold))
    else:
        if value >= good_threshold:
            return 1.0
        if value <= bad_threshold:
            return 0.0
        return float((value - bad_threshold) / (good_threshold - bad_threshold))
def map_docking_to_reward(docking_score: Optional[float]) -> float:
    return map_piecewise_to_reward(docking_score, good_threshold=-10.0, bad_threshold=-4.0, minimize=True)
def map_bioactivity_to_reward(bioactivity_score: Optional[float]) -> float:
    return map_piecewise_to_reward(bioactivity_score, good_threshold=-10.0, bad_threshold=-4.0, minimize=True)
def map_steric_to_reward(steric_score: Optional[float]) -> float:
    return map_piecewise_to_reward(steric_score, good_threshold=2.0, bad_threshold=15.0, minimize=True)
def map_torsion_to_reward(torsion_score: Optional[float]) -> float:
    return map_piecewise_to_reward(torsion_score, good_threshold=1.0, bad_threshold=5.0, minimize=True)
def map_qed_to_reward(qed_score: Optional[float]) -> float:
    return float(qed_score) if qed_score is not None else 0.0
def get_constraint_thresholds(curriculum_stage: int, valid_ratio: float = 1.0) -> Dict[str,float]:
    if curriculum_stage <= 1:
        base_thresholds = {
            'energy_max': 300.0,
            'sa_min': 3.0,
            'steric_max': 18.0,
            'torsion_max': 4.0
        }
    elif curriculum_stage == 2:
        base_thresholds = {
            'energy_max': 200.0,
            'sa_min': 2.7,
            'steric_max': 15.0,
            'torsion_max': 3.0
        }
    else:  # Stage 3
        base_thresholds = {
            'energy_max': 150.0,
            'sa_min': 2.5,
            'steric_max': 12.0,
            'torsion_max': 2.0
        }   
    if curriculum_stage >= 3 and valid_ratio < 0.30:
        relaxation_factor = 1.2
        logger.warning(f"Valid ratio {valid_ratio:.2%} < 30%, relaxing Stage 3 constraints by 20%")        
        return {
            'energy_max': base_thresholds['energy_max'] * relaxation_factor,
            'sa_min': base_thresholds['sa_min'] * 0.9,  # Slightly reduce requirement
            'steric_max': base_thresholds['steric_max'] * relaxation_factor,
            'torsion_max': base_thresholds['torsion_max'] * relaxation_factor
        }    
    return base_thresholds
def violates_hard_constraints(
    scores: Dict[str, Optional[float]],
    thresholds: Dict[str, float]
) -> bool:
    energy = scores.get('energy')
    if energy is not None and energy > thresholds['energy_max']:
        return True    
    sa = scores.get('sa')
    if sa is not None and sa < thresholds['sa_min']:
        return True    
    steric = scores.get('steric')
    if steric is not None and steric > thresholds['steric_max']:
        return True    
    torsion = scores.get('torsion')
    if torsion is not None and torsion > thresholds['torsion_max']:
        return True    
    return False
def get_curriculum_weights(curriculum_stage: int) -> Dict[str, float]:
    if curriculum_stage == 1:
        return {
            'docking': 0.35,
            'bioactivity': 0.35,
            'energy': 0.15,
            'sa': 0.05,
            'qed': 0.05,
            'steric': 0.03,
            'torsion': 0.02
        }
    elif curriculum_stage == 2:
        return {
            'docking': 0.30,
            'bioactivity': 0.30,
            'energy': 0.20,
            'sa': 0.08,
            'qed': 0.06,
            'steric': 0.04,
            'torsion': 0.02
        }
    else:
        return {
            'docking': 0.28,
            'bioactivity': 0.28,
            'energy': 0.22,
            'sa': 0.08,
            'qed': 0.06,
            'steric': 0.05,
            'torsion': 0.03
        }
def compute_base_reward_from_scores(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, float]
) -> float:
    reward_components = {
        'docking': map_docking_to_reward(scores.get('docking')),
        'bioactivity': map_bioactivity_to_reward(scores.get('bioactivity')),
        'energy': map_energy_to_reward(scores.get('energy')),
        'sa': map_sa_to_reward(scores.get('sa')),
        'qed': map_qed_to_reward(scores.get('qed')),
        'steric': map_steric_to_reward(scores.get('steric')),
        'torsion': map_torsion_to_reward(scores.get('torsion'))
    }    
    total_reward = sum(
        weights.get(name, 0.0) * reward
        for name, reward in reward_components.items()
    )    
    return total_reward
def convert_sentinel_to_optional(scores: Dict[str, float]) -> Dict[str, Optional[float]]:
    converted = {}
    for key, value in scores.items():
        if value is None or value == -1.0 or (isinstance(value, float) and np.isnan(value)):
            converted[key] = None
        else:
            converted[key] = value
    return converted
def convert_batch_to_optional(batch_scores: List[Dict[str, float]]) -> List[Dict[str, Optional[float]]]:
    return [convert_sentinel_to_optional(scores) for scores in batch_scores]
def has_valid_scores(scores: Dict[str, Optional[float]], objectives: List[Objective]) -> bool:
    return any(scores.get(obj.name) is not None for obj in objectives)
def compute_novelty_decay_factor(iteration: int, decay_iterations: int) -> float:
    return max(0.2, 1.0 - 0.8 * (iteration / decay_iterations))
def compute_fingerprint_from_smiles(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem        
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None        
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        return fingerprint
    except Exception:
        return None
def compute_fingerprint_hash(fingerprint) -> int:
    return hash(fingerprint.ToBitString())
def extract_murcko_scaffold(smiles: str) -> Optional[str]:
    if not HAS_MURCKO:
        return None    
    try:
        from rdkit import Chem
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None        
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        return scaffold if scaffold else None
    except Exception:
        return None
def compute_novelty_bonus_for_molecule(
    smiles: str,
    seen_fingerprints: List,
    seen_fp_hashes: set,
    seen_scaffolds: OrderedDict,
    iteration: int,
    config: OptimizationConfig
) -> Tuple[float, List, set, OrderedDict]:
    fingerprint = compute_fingerprint_from_smiles(smiles)
    if fingerprint is None:
        return 0.0, seen_fingerprints, seen_fp_hashes, seen_scaffolds    
    fp_hash = compute_fingerprint_hash(fingerprint)
    decay_factor = compute_novelty_decay_factor(iteration, config.novelty_decay_iterations)
    fp_weight = config.novelty_fp_weight_initial * decay_factor
    scaffold_weight = config.novelty_scaffold_bonus_initial * decay_factor    
    updated_fps = list(seen_fingerprints)
    updated_hashes = set(seen_fp_hashes)    
    if not updated_fps:
        updated_fps.append(fingerprint)
        updated_hashes.add(fp_hash)
        fp_novelty = 1.0
    else:
        min_similarity = compute_batch_tanimoto_minimum_similarity(
            fingerprint, 
            updated_fps, 
            max_references=100
        )
        fp_novelty = 1.0 - min_similarity        
        if fp_hash not in updated_hashes:
            updated_fps.append(fingerprint)
            updated_hashes.add(fp_hash)
    if len(updated_fps) > config.fingerprint_cache_size:
        updated_fps = updated_fps[-config.fingerprint_cache_size:]
        updated_hashes = {compute_fingerprint_hash(fp) for fp in updated_fps}    
    scaffold_bonus = 0.0
    updated_scaffolds = OrderedDict(seen_scaffolds)    
    scaffold = extract_murcko_scaffold(smiles)
    if scaffold and scaffold not in updated_scaffolds:
        scaffold_bonus = scaffold_weight        
        while len(updated_scaffolds) >= config.scaffold_cache_size:
            updated_scaffolds.popitem(last=False)        
        updated_scaffolds[scaffold] = True
    elif scaffold in updated_scaffolds:
        updated_scaffolds.move_to_end(scaffold)    
    total_novelty = fp_weight * fp_novelty + scaffold_bonus
    return total_novelty, updated_fps, updated_hashes, updated_scaffolds
def generate_uniform_weight_vectors_2d(num_vectors: int) -> List[np.ndarray]:
    return [
        np.array([i / (num_vectors - 1), 1 - i / (num_vectors - 1)])
        for i in range(num_vectors)
    ]
def generate_das_dennis_reference_points(num_objectives: int, divisions: int) -> List[np.ndarray]:
    all_weights = []    
    def generate_recursive(level: int, remaining: float, current_result: List[float]):
        if level == num_objectives - 1:
            current_result.append(remaining)
            all_weights.append(np.array(current_result.copy()))
            current_result.pop()
            return        
        for i in range(divisions + 1):
            weight = i / divisions
            if weight <= remaining + 1e-6:
                current_result.append(weight)
                generate_recursive(level + 1, remaining - weight, current_result)
                current_result.pop()    
    generate_recursive(0, 1.0, [])    
    if len(all_weights) > 50:
        indices = np.linspace(0, len(all_weights) - 1, 50, dtype=int)
        all_weights = [all_weights[i] for i in indices]    
    return all_weights
def generate_weight_vectors(num_objectives: int) -> List[np.ndarray]:
    if num_objectives == 2:
        return generate_uniform_weight_vectors_2d(5)
    elif num_objectives <= 4:
        return generate_das_dennis_reference_points(num_objectives, divisions=3)
    else:
        return generate_das_dennis_reference_points(num_objectives, divisions=2)
def normalize_scores_for_objectives(
    scores: Dict[str, Optional[float]],
    objectives: List[Objective]
) -> Dict[str, float]:
    normalized = {}    
    for objective in objectives:
        value = scores.get(objective.name)
        if value is not None:
            params = ObjectiveNormalizationParams(
                vmin=objective.vmin,
                vmax=objective.vmax,
                minimize=objective.minimize
            )
            norm_value = normalize_objective_value(value, params)
            if norm_value is not None:
                normalized[objective.name] = norm_value    
    return normalized
def compute_tchebycheff_utility_for_weights(
    normalized_scores: Dict[str, float],
    weights: np.ndarray,
    objectives: List[Objective],
    ideal_point: Dict[str, float]
) -> Tuple[float, float]:
    max_distance = 0.0
    weighted_sum = 0.0
    total_weight = 0.0
    valid_count = 0    
    for i, objective in enumerate(objectives):
        if objective.name not in normalized_scores:
            continue        
        normalized = normalized_scores[objective.name]
        weight = weights[i]        
        if weight < 1e-6:
            continue        
        ideal = ideal_point.get(objective.name, 1.0)
        distance = max(0, ideal - normalized)
        weighted_distance = weight * distance        
        max_distance = max(max_distance, weighted_distance)
        weighted_sum += weight * normalized
        total_weight += weight
        valid_count += 1    
    if valid_count == 0 or total_weight == 0:
        return -np.inf, 0.0    
    augmentation = 0.01 * (weighted_sum / total_weight)
    utility = 1.0 - max_distance + augmentation    
    return utility, max_distance
def compute_best_tchebycheff_utility(
    scores: Dict[str, Optional[float]],
    objectives: List[Objective],
    weight_vectors: List[np.ndarray],
    ideal_point: Dict[str, float],
    excellence_threshold: float,
    excellence_max_bonus: float
) -> float:
    normalized_scores = normalize_scores_for_objectives(scores, objectives)    
    if not normalized_scores:
        return 0.0    
    best_utility = -np.inf
    best_max_distance = np.inf    
    for weights in weight_vectors:
        utility, max_distance = compute_tchebycheff_utility_for_weights(
            normalized_scores,
            weights,
            objectives,
            ideal_point
        )        
        if utility > best_utility:
            best_utility = utility
            best_max_distance = max_distance    
    if best_utility == -np.inf:
        return 0.0    
    excellence_bonus = compute_smooth_excellence_bonus(
        best_max_distance,
        excellence_threshold,
        excellence_max_bonus
    )    
    return float(best_utility + excellence_bonus)
def check_dominance(
    solution_a: Dict[str, Optional[float]],
    solution_b: Dict[str, Optional[float]],
    objectives: List[Objective]
) -> int:
    a_better_count = 0
    b_better_count = 0    
    for objective in objectives:
        value_a = solution_a.get(objective.name)
        value_b = solution_b.get(objective.name)        
        if value_a is None or value_b is None:
            continue        
        params = ObjectiveNormalizationParams(
            vmin=objective.vmin,
            vmax=objective.vmax,
            minimize=objective.minimize
        )        
        norm_a = normalize_objective_value(value_a, params)
        norm_b = normalize_objective_value(value_b, params)        
        if norm_a is None or norm_b is None:
            continue        
        if norm_a > norm_b:
            a_better_count += 1
        elif norm_b > norm_a:
            b_better_count += 1    
    if a_better_count > 0 and b_better_count == 0:
        return 1
    elif b_better_count > 0 and a_better_count == 0:
        return -1
    else:
        return 0
def filter_dominated_solutions_from_archive(
    new_solution: Dict[str, Optional[float]],
    archive: List[Dict[str, Optional[float]]],
    objectives: List[Objective]
) -> Tuple[bool, List[Dict[str, Optional[float]]]]:
    is_new_dominated = False
    filtered_archive = []    
    for archived_solution in archive:
        dominance = check_dominance(new_solution, archived_solution, objectives)        
        if dominance == -1:
            is_new_dominated = True
            filtered_archive.append(archived_solution)
        elif dominance == 1:
            continue
        else:
            filtered_archive.append(archived_solution)    
    return is_new_dominated, filtered_archive
def compute_hypervolume_contributions(
    archive: List[Dict[str, Optional[float]]],
    objectives: List[Objective]
) -> np.ndarray:
    num_solutions = len(archive)
    num_objectives = len(objectives)    
    matrix = np.zeros((num_solutions, num_objectives))    
    for i, solution in enumerate(archive):
        for j, objective in enumerate(objectives):
            value = solution.get(objective.name)
            if value is not None:
                params = ObjectiveNormalizationParams(
                    vmin=objective.vmin,
                    vmax=objective.vmax,
                    minimize=objective.minimize
                )
                norm = normalize_objective_value(value, params)
                matrix[i, j] = norm if norm is not None else 0
            else:
                matrix[i, j] = 0    
    contributions = np.zeros(num_solutions)    
    for j in range(num_objectives):
        sorted_indices = np.argsort(matrix[:, j])[::-1]        
        contributions[sorted_indices[0]] += 1.0
        contributions[sorted_indices[-1]] += 0.3        
        for rank in range(1, len(sorted_indices) - 1):
            idx = sorted_indices[rank]
            prev_val = matrix[sorted_indices[rank - 1], j]
            next_val = matrix[sorted_indices[rank + 1], j]
            gap = prev_val - next_val
            contributions[idx] += gap    
    return contributions
def prune_archive_by_hypervolume(
    archive: List[Dict[str, Optional[float]]],
    objectives: List[Objective],
    target_size: int
) -> List[Dict[str, Optional[float]]]:
    if len(archive) <= target_size:
        return archive    
    contributions = compute_hypervolume_contributions(archive, objectives)
    kept_indices = np.argsort(contributions)[-target_size:]    
    return [archive[i] for i in sorted(kept_indices)]
def archive_contains_smiles(archive: List[Dict[str, Optional[float]]], smiles: Optional[str]) -> bool:
    if smiles is None:
        return False
    return any(solution.get('smiles') == smiles for solution in archive)
def compute_solution_utility(
    solution: Dict[str, Optional[float]],
    weights: Dict[str, float]
) -> float:
    utility_components = {
        'docking': map_docking_to_reward(solution.get('docking')),
        'bioactivity': map_bioactivity_to_reward(solution.get('bioactivity')),
        'energy': map_energy_to_reward(solution.get('energy')),
        'sa': map_sa_to_reward(solution.get('sa')),
        'qed': map_qed_to_reward(solution.get('qed')),
        'steric': map_steric_to_reward(solution.get('steric')),
        'torsion': map_torsion_to_reward(solution.get('torsion'))
    }    
    return sum(weights.get(name, 0.0) * value for name, value in utility_components.items())
def find_best_batch_candidate(
    batch_scores: List[Dict[str, Optional[float]]],
    smiles_list: Optional[List[str]],
    objectives: List[Objective],
    weights: Dict[str, float],
    thresholds: Dict[str, float],
    archive: List[Dict[str, Optional[float]]],
    archive_size: int
) -> Optional[Tuple[int, Dict[str, Optional[float]]]]:
    candidates = []    
    for i, scores in enumerate(batch_scores):
        if not has_valid_scores(scores, objectives):
            continue        
        if violates_hard_constraints(scores, thresholds):
            continue        
        utility = compute_solution_utility(scores, weights)
        candidates.append((i, utility, scores))    
    if not candidates:
        return None    
    candidates.sort(key=lambda x: x[1], reverse=True)    
    archive_best_utility = None
    if archive:
        archive_best_utility = max(
            compute_solution_utility(sol, weights) for sol in archive
        )    
    for idx, utility, scores in candidates:
        smiles = smiles_list[idx] if smiles_list and idx < len(smiles_list) else None        
        if archive_contains_smiles(archive, smiles):
            continue        
        if archive_best_utility is not None and len(archive) >= archive_size:
            if utility <= archive_best_utility + 1e-12:
                continue        
        return idx, scores    
    return None
class MultiObjectiveOptimizer:
    def __init__(self, batch_size: int = 32, archive_size: int = 100):
        self.config = OptimizationConfig(batch_size=batch_size, archive_size=archive_size)
        self.objectives = create_objectives()
        self.stats = {obj.name: create_running_statistics() for obj in self.objectives}        
        self.iteration = 0
        self.curriculum_stage = 1
        self.pareto_archive = []
        self.reward_history = deque(maxlen=100)
        self.pareto_ratio_history = deque(maxlen=100)
        self.valid_ratio_history = deque(maxlen=100)
        self.improvement_window = deque(maxlen=50)
        self.curriculum_hysteresis_count = 0
        self.stage_transition_warmup = 0
        self.reward_baseline = 0.0  
        self.baseline_momentum = 0.9        
        self.seen_fingerprints = []
        self.seen_fp_hashes = set()
        self.seen_scaffolds = OrderedDict()        
        self.weight_vectors_cache = {}
        self.ideal_point = {}        
        self.single_pareto_per_batch = True
        self.positive_replay_buffer = deque(maxlen=100) 
        self.current_stage = 1        
        num_objectives = len(self.objectives)
        logger.info(f"Initialized optimizer with {num_objectives} objectives")
        logger.info("3-Stage Progressive Curriculum: Docking+Bio (0-200) → +QED+SA (200-400) → Full RL (400-700)")    
    def update(
        self,
        batch_scores: List[Dict[str, float]],
        valid_ratio: float = 0.0,
        smiles_list: Optional[List[str]] = None,
        failure_types: Optional[List[str]] = None,
        diversity_penalties: Optional[List[float]] = None,
        seqs_list: Optional[List] = None,
        nums_list: Optional[List] = None
    ) -> np.ndarray:
        self.iteration += 1        
        optional_scores = convert_batch_to_optional(batch_scores)
        self.valid_ratio_history.append(valid_ratio)        
        old_stage = self.curriculum_stage
        self._update_curriculum()        
        if self.curriculum_stage != old_stage:
            self._reset_stage_history()        
        for scores in optional_scores:
            for obj in self.objectives:
                if obj.name in scores:
                    value = scores.get(obj.name)
                    self.stats[obj.name] = update_running_statistics(self.stats[obj.name], value)       
        rewards = self._compute_batch_rewards(
            optional_scores,
            smiles_list,
            failure_types,
            diversity_penalties
        )
        if seqs_list and nums_list and smiles_list:
            for i, advantage in enumerate(rewards):
                if advantage > 0 and i < len(seqs_list) and i < len(nums_list):
                    self.positive_replay_buffer.append({
                        'smiles': smiles_list[i],
                        'seqs': seqs_list[i],
                        'numbers': nums_list[i],
                        'advantage': float(advantage),
                        'scores': optional_scores[i] if i < len(optional_scores) else {}
                    })        
        if self.single_pareto_per_batch:
            self._select_and_add_best_candidate(
                optional_scores,
                smiles_list,
                seqs_list,
                nums_list
            )        
        self.reward_history.append(float(np.mean(rewards)))
        self.pareto_ratio_history.append(0.0)        
        if len(self.reward_history) >= 2:
            improvement = list(self.reward_history)[-1] - list(self.reward_history)[-2]
            self.improvement_window.append(improvement)
        if len(self.positive_replay_buffer) > 100:
            self.positive_replay_buffer = sorted(
                self.positive_replay_buffer, 
                key=lambda x: x.get('advantage', 0.0), 
                reverse=True
            )[:100]
        if len(self.reward_history) > 200:
            self.reward_history = deque(list(self.reward_history)[-200:], maxlen=200)
        if len(self.pareto_ratio_history) > 200:
            self.pareto_ratio_history = deque(list(self.pareto_ratio_history)[-200:], maxlen=200)
        if len(self.valid_ratio_history) > 200:
            self.valid_ratio_history = deque(list(self.valid_ratio_history)[-200:], maxlen=200)
        if len(self.improvement_window) > 50:
            self.improvement_window = deque(list(self.improvement_window)[-50:], maxlen=50)        
        if self.iteration % 50 == 0:
            self._log_progress(rewards, optional_scores)
            self.update_empirical_bounds()        
        if self.iteration % 10 == 0:
            self.update_ideal_point()        
        if self.stage_transition_warmup > 0:
            self.stage_transition_warmup -= 1        
        return rewards    
    def update_empirical_bounds(self):
        for i, obj in enumerate(self.objectives):
            if not obj.dynamic_bounds:
                continue            
            stats = self.stats.get(obj.name)
            if not has_sufficient_data(stats, min_samples=100):
                continue            
            p5 = compute_statistics_percentile(stats, 5.0)
            p95 = compute_statistics_percentile(stats, 95.0)            
            new_vmin, new_vmax = compute_updated_bounds(
                obj.vmin,
                obj.vmax,
                p5,
                p95,
                obj.minimize,
                obj.bounds_momentum,
                obj.observation_frequency
            )            
            updated_obj = replace(obj, vmin=new_vmin, vmax=new_vmax)
            self.objectives[i] = updated_obj            
            logger.info(f"Updated {obj.name} bounds: [{new_vmin:.3f}, {new_vmax:.3f}]")    
    def update_ideal_point(self):
        if not self.pareto_archive:
            return        
        for obj in self.objectives:
            normalized_vals = []
            for solution in self.pareto_archive:
                val = solution.get(obj.name)
                if val is not None:
                    params = ObjectiveNormalizationParams(
                        vmin=obj.vmin, 
                        vmax=obj.vmax, 
                        minimize=obj.minimize
                    )
                    norm = normalize_objective_value(val, params)
                    if norm is not None:
                        normalized_vals.append(norm)            
            if normalized_vals:
                current_ideal = self.ideal_point.get(obj.name, 0.5)
                new_ideal = max(normalized_vals)
                self.ideal_point[obj.name] = 0.9 * current_ideal + 0.1 * new_ideal    
    def _compute_batch_rewards(
        self,
        batch_scores: List[Dict[str, Optional[float]]],
        smiles_list: Optional[List[str]],
        failure_types: Optional[List[str]],
        diversity_penalties: Optional[List[float]]
    ) -> np.ndarray:
        rewards = np.zeros(len(batch_scores))
        weights = get_curriculum_weights(self.curriculum_stage)
        recent_valid_ratio = float(np.mean(list(self.valid_ratio_history)[-20:])) if len(self.valid_ratio_history) >= 20 else 1.0
        thresholds = get_constraint_thresholds(self.curriculum_stage, recent_valid_ratio)        
        for i, scores in enumerate(batch_scores):
            if not has_valid_scores(scores, self.objectives):
                rewards[i] = -1.0
                continue            
            if violates_hard_constraints(scores, thresholds):
                rewards[i] = -1.0
                continue            
            base_reward = compute_base_reward_from_scores(scores, weights)            
            base_reward = float(np.clip(base_reward, -2.0, 2.0))
            diversity_penalty = 0.0
            if smiles_list and i < len(smiles_list):
                smiles_count = smiles_list.count(smiles_list[i])
                if base_reward > 0.3 and smiles_count > 1:
                    diversity_penalty = -0.01 * (smiles_count - 1)
            novelty_bonus = 0.0
            if smiles_list and i < len(smiles_list) and base_reward > 0.3 and diversity_penalty == 0.0:
                novelty_bonus, self.seen_fingerprints, self.seen_fp_hashes, self.seen_scaffolds = (
                    compute_novelty_bonus_for_molecule(
                        smiles_list[i],
                        self.seen_fingerprints,
                        self.seen_fp_hashes,
                        self.seen_scaffolds,
                        self.iteration,
                        self.config
                    )
                )
                novelty_bonus = min(novelty_bonus, 0.1 * base_reward)
            raw_reward = base_reward + diversity_penalty + novelty_bonus
            rewards[i] = float(np.clip(raw_reward, -1.0, 1.0))     
        valid_rewards = rewards[rewards > -0.9]        
        if len(valid_rewards) > 0:
            valid_mean = float(np.mean(valid_rewards))
            self.reward_baseline = self.baseline_momentum * self.reward_baseline + (1 - self.baseline_momentum) * valid_mean
        advantages = rewards - self.reward_baseline
        advantages = np.clip(advantages, -3.0, 3.0)        
        return advantages    
    def update_curriculum_stage(self, current_step):
        old_stage = self.current_stage        
        if current_step < 200:
            self.current_stage = 1  
            stage_name = "Docking+Bio Focus"
        elif current_step < 400:
            self.current_stage = 2 
            stage_name = "Docking+Bio+QED+SA"
        else:
            self.current_stage = 3  
            stage_name = "Full Multi-Objective"        
        if old_stage != self.current_stage:
            logger.info(f"Curriculum advanced to Stage {self.current_stage} at step {current_step}: {stage_name}")
            if self.current_stage == 2:
                logger.info(f"Stage 2: Adding QED + SA optimization")
            elif self.current_stage == 3:
                logger.info(f"Stage 3: Adding Docking + Bioactivity")
            elif self.current_stage == 4:
                logger.info(f"Stage 4: Full multi-objective RL")
            logger.info(f"{'='*60}\n")
            self.reward_baseline = 0.0      
    def _compute_single_reward(self, scores: Dict[str, float], smiles: Optional[str] = None) -> float:
        optional_scores = convert_sentinel_to_optional(scores)
        if self.current_stage == 1:
            is_valid = any(
                optional_scores.get(prop) is not None 
                for prop in ['qed', 'sa', 'steric', 'torsion', 'energy']
            )            
            if is_valid:
                all_basic_valid = all(
                    optional_scores.get(prop) is not None 
                    for prop in ['qed', 'sa', 'steric', 'torsion', 'energy']
                )
                return 1.0 if all_basic_valid else 0.5
            else:
                return -1.0
        elif self.current_stage == 2:
            is_valid = any(
                optional_scores.get(prop) is not None 
                for prop in ['qed', 'sa', 'steric', 'torsion', 'energy']
            )            
            if not is_valid:
                return -1.0
            reward = 0.5
            qed = optional_scores.get('qed')
            sa = optional_scores.get('sa')            
            if qed is not None:
                qed_bonus = qed * 0.3 
                reward += qed_bonus            
            if sa is not None:
                sa_normalized = 1.0 - ((sa - 1.0) / 9.0)
                sa_bonus = sa_normalized * 0.3  
                reward += sa_bonus
            steric = optional_scores.get('steric')
            if steric is not None and steric < 5.0:
                steric_bonus = 0.1
                reward += steric_bonus            
            return float(np.clip(reward, -1.0, 1.0))
        elif self.current_stage == 3:
            is_valid = any(
                optional_scores.get(prop) is not None 
                for prop in ['qed', 'sa', 'steric', 'torsion', 'energy']
            )            
            if not is_valid:
                return -1.0
            reward = 0.5
            qed = optional_scores.get('qed')
            sa = optional_scores.get('sa')            
            if qed is not None:
                qed_bonus = qed * 0.2
                reward += qed_bonus            
            if sa is not None:
                sa_normalized = 1.0 - ((sa - 1.0) / 9.0)
                sa_bonus = sa_normalized * 0.2
                reward += sa_bonus            
            docking = optional_scores.get('docking')
            bioactivity = optional_scores.get('bioactivity')            
            if docking is not None:
                docking_normalized = (max(-12, min(-6, docking)) + 12) / 6.0
                docking_bonus = docking_normalized * 0.3  
                reward += docking_bonus            
            if bioactivity is not None:
                bioactivity_bonus = bioactivity * 0.3  
                reward += bioactivity_bonus            
            return float(np.clip(reward, -1.0, 1.0))        
        elif self.current_stage == 4: 
            is_valid = any(
                optional_scores.get(prop) is not None 
                for prop in ['qed', 'sa', 'steric', 'torsion', 'energy']
            )            
            if not is_valid:
                return -1.0
            weights = get_curriculum_weights(self.curriculum_stage)
            recent_valid_ratio = float(np.mean(list(self.valid_ratio_history)[-20:])) if len(self.valid_ratio_history) >= 20 else 1.0
            thresholds = get_constraint_thresholds(self.curriculum_stage, recent_valid_ratio)
            if violates_hard_constraints(optional_scores, thresholds):
                return 0.0             
            base_reward = compute_base_reward_from_scores(optional_scores, weights)
            base_reward = float(np.clip(base_reward, -2.0, 2.0)) 
            advantage = base_reward - self.reward_baseline
            return float(np.clip(advantage, -1.0, 1.0))    
    def _select_and_add_best_candidate(
        self,
        batch_scores: List[Dict[str, Optional[float]]],
        smiles_list: Optional[List[str]],
        seqs_list: Optional[List],
        nums_list: Optional[List]
    ):
        weights = get_curriculum_weights(self.curriculum_stage)
        thresholds = get_constraint_thresholds(self.curriculum_stage)        
        result = find_best_batch_candidate(
            batch_scores,
            smiles_list,
            self.objectives,
            weights,
            thresholds,
            self.pareto_archive,
            self.config.archive_size
        )        
        if result is None:
            return        
        idx, scores = result        
        entry = scores.copy()
        if smiles_list and idx < len(smiles_list):
            entry['smiles'] = smiles_list[idx]
        if seqs_list and idx < len(seqs_list):
            entry['seqs'] = seqs_list[idx]
        if nums_list and idx < len(nums_list):
            entry['numbers'] = nums_list[idx]        
        is_dominated, filtered_archive = filter_dominated_solutions_from_archive(
            entry,
            self.pareto_archive,
            self.objectives
        )        
        if not is_dominated:
            filtered_archive.append(entry)
            self.pareto_archive = filtered_archive            
            if len(self.pareto_archive) > self.config.archive_size:
                self.pareto_archive = prune_archive_by_hypervolume(
                    self.pareto_archive,
                    self.objectives,
                    self.config.archive_size
                )            
            logger.info(f"Added solution to archive (size: {len(self.pareto_archive)})")    
    def _update_curriculum(self):
        pass    
    def _reset_stage_history(self):
        self.pareto_ratio_history.clear()
        self.improvement_window.clear()
        self.stage_transition_warmup = 50
        for obj in self.objectives:
            self.ideal_point[obj.name] = 0.5
        if len(self.pareto_archive) > 20:
            best = self.get_best_solutions(top_k=20)
            self.pareto_archive = list(best)
            logger.info(f"Pruned archive from {len(self.pareto_archive)} to 20 best solutions for new curriculum stage")        
        logger.info(f"Reset history for curriculum stage {self.curriculum_stage}")    
    def _log_progress(self, rewards: np.ndarray, batch_scores: List[Dict[str, Optional[float]]]):
        valid_count = sum(1 for scores in batch_scores if has_valid_scores(scores, self.objectives))
        total = len(batch_scores)        
        avg_improvement = float(np.mean(list(self.improvement_window))) if self.improvement_window else 0.0        
        logger.info(
            f"Iter {self.iteration:4d} | "
            f"Stage {self.curriculum_stage} | "
            f"Valid {valid_count}/{total} | "
            f"Reward {np.mean(rewards):.4f}±{np.std(rewards):.4f} | "
            f"Improvement {avg_improvement:+.6f} | "
            f"Archive {len(self.pareto_archive)}"
        )    
    def get_best_solutions(self, top_k: int = 10) -> List[Dict[str, Optional[float]]]:
        if not self.pareto_archive:
            return []        
        weight_vectors = self._get_weight_vectors(len(self.objectives))        
        scored = []
        for solution in self.pareto_archive:
            score = compute_best_tchebycheff_utility(
                solution,
                self.objectives,
                weight_vectors,
                self.ideal_point,
                self.config.excellence_threshold,
                self.config.excellence_max_bonus
            )
            scored.append((solution, score))        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [solution for solution, _ in scored[:top_k]]    
    def get_experience_replay_batch(self, top_k: int = 32) -> List[Dict[str, Optional[float]]]:
        if not self.positive_replay_buffer:
            best_solutions = self.get_best_solutions(top_k)
            return [dict(solution) for solution in best_solutions]
        buffer_list = list(self.positive_replay_buffer)
        buffer_list.sort(key=lambda x: x.get('advantage', 0), reverse=True)
        num_samples = min(top_k, len(buffer_list))
        return buffer_list[:num_samples]    
    def _get_weight_vectors(self, num_objectives: int) -> List[np.ndarray]:
        if num_objectives in self.weight_vectors_cache:
            return self.weight_vectors_cache[num_objectives]        
        vectors = generate_weight_vectors(num_objectives)
        self.weight_vectors_cache[num_objectives] = vectors
        return vectors    
    def get_metrics(self) -> Dict[str, float]:
        metrics = {
            'mean_reward': 0.0,
            'reward_std': 0.0,
            'pareto_ratio': 0.0,
            'archive_size': len(self.pareto_archive),
            'curriculum_stage': self.curriculum_stage,
            'novelty_cache_size': len(self.seen_fingerprints),
            'improvement_rate': float(np.mean(list(self.improvement_window))) if self.improvement_window else 0.0,
        }        
        if self.reward_history:
            rewards = list(self.reward_history)
            metrics['mean_reward'] = float(np.mean(rewards))
            metrics['reward_std'] = float(np.std(rewards))        
        if self.pareto_ratio_history:
            metrics['pareto_ratio'] = float(np.mean(list(self.pareto_ratio_history)))        
        return metrics    
    def state_dict(self) -> Dict:
        return {
            'iteration': self.iteration,
            'curriculum_stage': self.curriculum_stage,
            'curriculum_hysteresis_count': self.curriculum_hysteresis_count,
            'stage_transition_warmup': self.stage_transition_warmup,
            'pareto_archive': list(self.pareto_archive),
            'reward_history': list(self.reward_history),
            'pareto_ratio_history': list(self.pareto_ratio_history),
            'valid_ratio_history': list(self.valid_ratio_history),
            'improvement_window': list(self.improvement_window),
            'seen_scaffolds': list(self.seen_scaffolds.keys()),
            'ideal_point': dict(self.ideal_point),
        }    
    def load_state_dict(self, state: Dict):
        self.iteration = state.get('iteration', 0)
        self.curriculum_stage = state.get('curriculum_stage', 1)
        self.curriculum_hysteresis_count = state.get('curriculum_hysteresis_count', 0)
        self.stage_transition_warmup = state.get('stage_transition_warmup', 0)
        self.pareto_archive = state.get('pareto_archive', [])
        self.ideal_point = state.get('ideal_point', {})        
        self.reward_history = deque(state.get('reward_history', []), maxlen=100)
        self.pareto_ratio_history = deque(state.get('pareto_ratio_history', []), maxlen=100)
        self.valid_ratio_history = deque(state.get('valid_ratio_history', []), maxlen=100)
        self.improvement_window = deque(state.get('improvement_window', []), maxlen=50)        
        scaffolds = state.get('seen_scaffolds', [])
        self.seen_scaffolds = OrderedDict((s, True) for s in scaffolds)        
        logger.info(f"Loaded optimizer state: iteration={self.iteration}, stage={self.curriculum_stage}")