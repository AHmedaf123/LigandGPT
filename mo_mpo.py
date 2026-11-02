import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, replace
from collections import Counter, deque, OrderedDict
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
def resolve_stage(step: int, n_steps: int) -> int:
    """The only place the curriculum stage is decided: equal thirds, boundaries n_steps//3 and 2*n_steps//3.
    Weights, gates, archive selection and metrics all call this, so the stage cannot differ between them."""
    return 1 if step < n_steps // 3 else (2 if step < 2 * n_steps // 3 else 3)
def get_constraint_thresholds(curriculum_stage: int, valid_ratio: float = 1.0) -> Dict[str,float]:
    if curriculum_stage <= 1:
        base_thresholds = {
            'energy_max': 300.0,
            'sa_max': 7.575,   # SA gate: reject sa > P90 / P80 / P70 of the valid baseline molecules
            'steric_max': 18.0,
            'torsion_max': 5.0    # torsion 5/4/3 and energy 300/200/150 are reconstructed values (the originals were lost)
        }
    elif curriculum_stage == 2:
        base_thresholds = {
            'energy_max': 200.0,
            'sa_max': 6.805,   # (runs/baseline/baseline.json, 93 valid molecules, structured decoder, targets 0 and 3)
            'steric_max': 15.0,
            'torsion_max': 4.0
        }
    else:  # Stage 3
        base_thresholds = {
            'energy_max': 150.0,
            'sa_max': 6.254,
            'steric_max': 12.0,
            'torsion_max': 3.0
        }   
    if curriculum_stage >= 3 and valid_ratio < 0.15:
        relaxation_factor = 1.2
        logger.warning(f"Valid ratio {valid_ratio:.2%} < 15%, relaxing Stage 3 constraints by 20%")        
        return {
            'energy_max': base_thresholds['energy_max'] * relaxation_factor,
            'sa_max': base_thresholds['sa_max'] * 1.1,  # relaxing allows a higher (worse) SA
            'steric_max': base_thresholds['steric_max'] * relaxation_factor,
            'torsion_max': base_thresholds['torsion_max'] * relaxation_factor
        }    
    return base_thresholds
def is_unscored(scores: Dict[str, Optional[float]]) -> bool:
    """A valid molecule whose 3D build failed: no steric/torsion/energy score. (A failed docking is handled by the RL loop.)"""
    return any(scores.get(key) is None for key in ('steric', 'torsion', 'energy'))
def violates_hard_constraints(
    scores: Dict[str, Optional[float]],
    thresholds: Dict[str, float]
) -> bool:
    if is_unscored(scores):
        return True
    energy = scores.get('energy')
    if energy is not None and energy > thresholds['energy_max']:
        return True    
    sa = scores.get('sa')
    if sa is not None and sa > thresholds['sa_max']:
        return True    
    steric = scores.get('steric')
    if steric is not None and steric > thresholds['steric_max']:
        return True    
    torsion = scores.get('torsion')
    if torsion is not None and torsion > thresholds['torsion_max']:
        return True    
    return False
def get_curriculum_weights(curriculum_stage: int) -> Dict[str, float]:
    # steric has weight 0 in every stage (the steric gate still applies). The equal-thirds average of the three vectors is
    # docking 0.450, bioactivity 0.097, QED 0.093, SA 0.093, torsion 0.080, energy 0.187 = the no-curriculum control.
    if curriculum_stage == 1:
        return {
            'docking': 0.60,
            'bioactivity': 0.06,
            'energy': 0.14,
            'sa': 0.07,
            'qed': 0.07,
            'steric': 0.0,
            'torsion': 0.06
        }
    elif curriculum_stage == 2:
        return {
            'docking': 0.40,
            'bioactivity': 0.14,
            'energy': 0.14,
            'sa': 0.13,
            'qed': 0.13,
            'steric': 0.0,
            'torsion': 0.06
        }
    else:
        return {
            'docking': 0.35,
            'bioactivity': 0.09,
            'energy': 0.28,
            'sa': 0.08,
            'qed': 0.08,
            'steric': 0.0,
            'torsion': 0.12
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
