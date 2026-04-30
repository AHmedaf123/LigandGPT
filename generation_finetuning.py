import os
import argparse
import lmdb
import pickle
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from rdkit import Chem
import selfies as sf
from collections import Counter
import gc
from model import LigandGPT
from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary
from training_utils import predict_smiles
from scoring_function import calculate_scores
from mo_mpo import MultiObjectiveOptimizer
def get_lmdb_data(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    return txn, keys
def calculate_pareto_frontier(memory_dataframe, top_n_molecules=100):
    if memory_dataframe.empty:
        return memory_dataframe    
    valid_molecules = memory_dataframe[memory_dataframe['docking'] != -1.0].copy()  
    if valid_molecules.empty:
        return memory_dataframe.head(top_n_molecules)    
    if len(valid_molecules) > 500:
        valid_molecules = valid_molecules.head(500)    
    objectives = valid_molecules[['docking', 'qed', 'sa', 'steric', 'torsion', 'energy', 'bioactivity']].values.astype(np.float32)
    objectives_transformed = objectives.copy()
    objectives_transformed[:, 2] = -objectives[:, 2]
    objectives_transformed[:, 3] = -objectives[:, 3]
    objectives_transformed[:, 4] = -objectives[:, 4]
    objectives_transformed[:, 5] = -objectives[:, 5]    
    n_molecules = len(objectives_transformed)
    pareto_mask = np.ones(n_molecules, dtype=bool)    
    for i in range(n_molecules):
        if not pareto_mask[i]:
            continue
        for j in range(n_molecules):
            if i != j and pareto_mask[j]:
                if (np.all(objectives_transformed[j] >= objectives_transformed[i]) and 
                    np.any(objectives_transformed[j] > objectives_transformed[i])):
                    pareto_mask[i] = False
                    break    
    pareto_frontier_dataframe = valid_molecules[pareto_mask].copy()
    pareto_frontier_dataframe = pareto_frontier_dataframe.sort_values('scores', ascending=False)   
    return pareto_frontier_dataframe.head(top_n_molecules)
def update_molecule_memory(current_memory, new_smiles_list, new_rewards_list, new_score_dicts_list, new_seqs_list, new_nums_list, memory_size_limit):
    new_data_list = []
    for i in range(len(new_smiles_list)):
        new_data_list.append({
            "smiles": new_smiles_list[i],
            "scores": new_rewards_list[i],
            "docking": new_score_dicts_list[i].get('docking', -1.0),
            "qed": new_score_dicts_list[i].get('qed', -1.0),
            "sa": new_score_dicts_list[i].get('sa', -1.0),
            "steric": new_score_dicts_list[i].get('steric', -1.0),
            "torsion": new_score_dicts_list[i].get('torsion', -1.0),
            "energy": new_score_dicts_list[i].get('energy', -1.0),
            "bioactivity": new_score_dicts_list[i].get('bioactivity', -1.0),
            "seqs": [new_seqs_list[i]],
            "numbers": [new_nums_list[i]]
        })    
    if new_data_list:
        new_molecules_dataframe = pd.DataFrame(new_data_list)
        combined_memory = pd.concat([current_memory, new_molecules_dataframe], ignore_index=True, sort=False)
    else:
        combined_memory = current_memory

    unique_memory = combined_memory.drop_duplicates(subset=["smiles"])
    sorted_memory = unique_memory.sort_values('scores', ascending=False)
    reset_memory = sorted_memory.reset_index(drop=True)    
    return reset_memory.head(memory_size_limit)
def generate_smiles_batch(agent_model, encoded_input_batch, numbers_input_batch, current_step, total_steps, debug_enabled, temperature_override=None, top_k_override=None):
    cosine_progress = 0.5 * (1 + np.cos(np.pi * current_step / total_steps))
    temperature = temperature_override if temperature_override is not None else (0.7 + 0.2 * cosine_progress)
    top_k = top_k_override if top_k_override is not None else max(30, int(70 - 40 * (1 - cosine_progress)))
    agent_model.eval()
    with torch.no_grad():
        predicted_encoded_sequences, predicted_numbers_sequences = predict_smiles(
            agent_model, encoded_input_batch, numbers_input_batch, max_len=512, 
            temperature=temperature, top_k=top_k, 
            debug=debug_enabled
        )
    agent_model.train()    
    return predicted_encoded_sequences, predicted_numbers_sequences, temperature, top_k
def validate_and_standardize_smiles(generated_encoded_sequences, generated_numbers_sequences, tokenizer_instance, vocabulary_instance):
    smiles_list = []
    valid_mask_list = []
    failure_types_list = []    
    for j in range(generated_encoded_sequences.shape[0]):
        predicted_encoded_j = generated_encoded_sequences[j, :].tolist()
        predicted_tokens_j = vocabulary_instance.decode(predicted_encoded_j)
        predicted_numbers_j = generated_numbers_sequences[j, :].tolist()
        current_smiles = ''
        is_valid = False
        failure_type = 'syntax'        
        try:
            sample_j = tokenizer_instance.untokenize(predicted_tokens_j, predicted_numbers_j)
            potential_smi = sample_j.get('LigSmiles', '')            
            if potential_smi:
                try:
                    selfies_str = sf.encoder(potential_smi)
                    fixed_smi = sf.decoder(selfies_str)
                    mol = Chem.MolFromSmiles(fixed_smi)
                    if mol is not None and mol.GetNumAtoms() >= 4:
                        current_smiles = Chem.MolToSmiles(mol, canonical=True)
                        is_valid = True
                        failure_type = 'valid'
                    elif mol is not None:
                        current_smiles = Chem.MolToSmiles(mol, canonical=True)
                        failure_type = 'chemistry'
                    else:
                        current_smiles = fixed_smi
                        failure_type = 'chemistry'
                except Exception:
                    mol = Chem.MolFromSmiles(potential_smi)
                    if mol is not None and mol.GetNumAtoms() >= 4:
                        current_smiles = Chem.MolToSmiles(mol, canonical=True)
                        is_valid = True
                        failure_type = 'valid'
                    elif mol is not None:
                        current_smiles = Chem.MolToSmiles(mol, canonical=True)
                        failure_type = 'chemistry'
                    else:
                        current_smiles = potential_smi
                        failure_type = 'syntax'
            else:
                failure_type = 'syntax'
        except Exception:
            failure_type = 'syntax'
        
        smiles_list.append(current_smiles)
        valid_mask_list.append(is_valid)
        failure_types_list.append(failure_type)        
    return smiles_list, valid_mask_list, failure_types_list
def prepare_experience_replay(mo_optimizer_instance, current_sequence_length, num_replay_molecules):
    if num_replay_molecules > 0 and len(mo_optimizer_instance.pareto_archive) > 0:
        er_seqs = []
        er_nums = []
        er_valid_mask = []
        er_smiles = []
        er_scores = []
        er_score_dicts = []        
        er_solutions = mo_optimizer_instance.get_experience_replay_batch(top_k=num_replay_molecules)       
        for sol in er_solutions:
            if 'seqs' not in sol or 'numbers' not in sol:
                continue
            try:
                seq_np = np.array(sol['seqs'])
                nums_np = np.array(sol['numbers'])
                seq_t = torch.tensor(seq_np, dtype=torch.long)
                nums_t = torch.tensor(nums_np, dtype=torch.float)
                PAD_TOKEN_ID = 0
                if seq_t.shape[0] < current_sequence_length:
                    pad_len = current_sequence_length - seq_t.shape[0]
                    seq_t = torch.cat([seq_t, torch.full((pad_len,), PAD_TOKEN_ID, dtype=torch.long)])
                    nums_t = torch.cat([nums_t, torch.zeros(pad_len, dtype=torch.float)])
                elif seq_t.shape[0] > current_sequence_length:
                    seq_t = seq_t[:current_sequence_length]
                    nums_t = nums_t[:current_sequence_length]                
                er_seqs.append(seq_t.cuda())
                er_nums.append(nums_t.cuda())
                er_valid_mask.append(True)
                er_smiles.append(sol.get('smiles', ''))
                score_dict = {
                    'docking': sol.get('docking'),
                    'qed': sol.get('qed'),
                    'sa': sol.get('sa'),
                    'steric': sol.get('steric'),
                    'torsion': sol.get('torsion'),
                    'energy': sol.get('energy'),
                    'bioactivity': sol.get('bioactivity')
                }
                er_reward = mo_optimizer_instance._compute_single_reward(score_dict, smiles=sol.get('smiles'))
                er_scores.append(er_reward)                
                er_score_dicts.append(score_dict)
            except Exception:
                continue        
        return er_seqs, er_nums, er_valid_mask, er_smiles, er_scores, er_score_dicts
    return [], [], [], [], [], []

def compute_reinforce_loss(agent_model, prior_model, all_encoded_sequences, all_numbers_sequences, advantages_tensor, entropy_coefficient):
    agent_model.eval()
    with torch.no_grad():
        prior_logits, _ = prior_model(all_encoded_sequences[:, :-1], all_numbers_sequences[:, :-1])
        prior_log_probabilities = torch.log_softmax(prior_logits, dim=-1)
        prior_token_log_probabilities = torch.gather(prior_log_probabilities, 2, all_encoded_sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        prior_likelihood = prior_token_log_probabilities.sum(dim=1)    
    agent_model.train()
    agent_logits, _ = agent_model(all_encoded_sequences[:, :-1], all_numbers_sequences[:, :-1])
    agent_log_probabilities = torch.log_softmax(agent_logits, dim=-1)
    agent_token_log_probabilities = torch.gather(agent_log_probabilities, 2, all_encoded_sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    agent_likelihood = agent_token_log_probabilities.sum(dim=1)
    kl_divergence_mean = (agent_likelihood - prior_likelihood).detach().mean().item()
    log_ratio = agent_likelihood - prior_likelihood.detach()
    log_ratio_clamped = torch.clamp(log_ratio, -10.0, 10.0)
    ratio = torch.exp(log_ratio_clamped)
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    unclipped_objective = ratio * advantages_tensor
    clipped_objective = clipped_ratio * advantages_tensor
    per_sample_loss = -torch.min(unclipped_objective, clipped_objective)
    weights = torch.ones_like(advantages_tensor)
    weights[advantages_tensor > 0] = 5.0
    policy_loss = (per_sample_loss * weights).sum() / weights.sum()
    entropy = -(torch.exp(agent_log_probabilities) * agent_log_probabilities).sum(dim=-1).mean()
    total_loss = policy_loss - entropy_coefficient * entropy
    total_loss = torch.clamp(total_loss, -10.0, 10.0)
    return total_loss, policy_loss, torch.tensor(0.0).cuda(), 0.0, entropy, kl_divergence_mean
def optimize_agent_parameters(agent_model, optimizer_instance, loss_value):
    if torch.isnan(loss_value) or torch.isinf(loss_value):
        print(f"WARNING: Invalid loss detected: {loss_value}")
        return False, 0.0    
    optimizer_instance.zero_grad()
    loss_value.backward()
    total_norm = 0.0
    for p in agent_model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_grad_norm_unclipped = total_norm ** 0.5
    total_grad_norm = torch.nn.utils.clip_grad_norm_(agent_model.parameters(), 5.0)    
    optimizer_instance.step()
    if total_grad_norm_unclipped > 10.0:
        print(f"WARNING: High gradient norm (before clip): {total_grad_norm_unclipped:.2f} (clipped to {total_grad_norm:.2f})")    
    return True, float(total_grad_norm)
def process_training_step(step_index, agent_model, prior_model, mo_optimizer_instance, optimizer_instance, initial_encoded_input, initial_numbers_input, receptor_file_path, ligand_center, training_arguments, current_memory_dataframe, tokenizer_instance, vocabulary_instance):
    mo_optimizer_instance.update_curriculum_stage(step_index)
    max_attempts = 3 if step_index == 0 else 1    
    for attempt in range(max_attempts):
        if attempt == 0:
            temp_override, topk_override = None, None
        elif attempt == 1:
            temp_override, topk_override = 1.2, 80
            print(f"  ⚠️  Attempt {attempt+1}/{max_attempts}: 0 valid molecules (mode collapse), retrying with temp=1.2, top_k=80...")
        else:
            temp_override, topk_override = 1.4, 80
            print(f"  ⚠️  Attempt {attempt+1}/{max_attempts}: 0 valid molecules (mode collapse), retrying with temp=1.4, top_k=80...")      
        generated_encoded_sequences, generated_numbers_sequences, temperature_value, top_k_value = generate_smiles_batch(
            agent_model, initial_encoded_input, initial_numbers_input, step_index, training_arguments.n_steps, 
            training_arguments.debug, temp_override, topk_override
        )        
        smiles_generated, valid_mask_generated, failure_types_generated = validate_and_standardize_smiles(
            generated_encoded_sequences, generated_numbers_sequences, tokenizer_instance, vocabulary_instance
        )        
        valid_count = sum(valid_mask_generated)
        if valid_count > 0:
            if attempt > 0:
                print(f"  ✓ Retry successful! Generated {valid_count}/16 valid molecules")
            break
    invalid_count = len(valid_mask_generated) - valid_count
    print(f"Step {step_index}: Valid={valid_count}/{len(valid_mask_generated)}, Invalid={invalid_count}")
    if step_index == 0 and invalid_count > 0:
        from collections import Counter
        failure_counts = Counter(failure_types_generated)
        print(f"  Failure breakdown: {dict(failure_counts)}")
        invalid_smiles = [sm for sm, valid in zip(smiles_generated, valid_mask_generated) if not valid][:3]
        if invalid_smiles:
            print(f"  Sample invalid SMILES: {invalid_smiles}")    
    raw_scores_evaluated = calculate_scores(smiles_generated, receptor_file=receptor_file_path, box_center=ligand_center)
    score_dicts_evaluated = [{'docking': s[0], 'qed': s[1], 'sa': s[2], 'steric': s[3], 'torsion': s[4], 'energy': s[5], 'bioactivity': s[6]} for s in raw_scores_evaluated]
    seqs_list_for_optimizer = [generated_encoded_sequences[i, :].detach().cpu().numpy() for i in range(training_arguments.batch_size)]
    nums_list_for_optimizer = [generated_numbers_sequences[i, :].detach().cpu().numpy() for i in range(training_arguments.batch_size)]    
    advantages_from_optimizer = mo_optimizer_instance.update(
        score_dicts_evaluated, 
        valid_ratio=sum(valid_mask_generated)/training_arguments.batch_size,
        smiles_list=smiles_generated,
        failure_types=failure_types_generated,
        diversity_penalties=None,  
        seqs_list=seqs_list_for_optimizer,
        nums_list=nums_list_for_optimizer
    )    
    if step_index > 0 and step_index % 50 == 0:
        mo_optimizer_instance.update_empirical_bounds()
    advantages = torch.tensor(np.array(advantages_from_optimizer), dtype=torch.float).cuda()
    if torch.isnan(advantages).any() or torch.isinf(advantages).any():
        print(f"WARNING: NaN or Inf detected in advantages, replacing with zeros")
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    advantages = torch.clamp(advantages, -5.0, 5.0)
    entropy_coefficient_value = 0.05 + 0.10 * (0.5 * (1 + np.cos(np.pi * step_index / training_arguments.n_steps)))
    valid_indices = [i for i, is_valid in enumerate(valid_mask_generated) if is_valid]
    if len(valid_indices) == 0:
        print(f"⚠️  Step {step_index}: NO valid molecules generated, skipping PPO update")
        updated_memory_dataframe = update_molecule_memory(
            current_memory_dataframe, smiles_generated, advantages_from_optimizer, score_dicts_evaluated, seqs_list_for_optimizer, nums_list_for_optimizer, training_arguments.memory_size
        )
        return (
            agent_model, 
            mo_optimizer_instance,
            updated_memory_dataframe, 
            0.0,  # total_loss
            0.0,  # policy_loss
            0.0,  # kl_penalty
            0.0,  # kl_coefficient
            0.0,  # entropy
            entropy_coefficient_value, 
            temperature_value, 
            0.0,  # kl_divergence_mean
            0.0,  # advantages.mean()
            0.0,  # advantages.std()
            0.0,  # valid_ratio
            smiles_generated,
            score_dicts_evaluated,
            advantages.cpu().numpy(),
            0.0   # grad_norm
        )
    print(f"✓ PPO will train on {len(valid_indices)} valid molecules (filtered out {invalid_count} invalid)")
    valid_encoded_seqs = generated_encoded_sequences[valid_indices]
    valid_numbers_seqs = generated_numbers_sequences[valid_indices]
    valid_advantages = torch.tensor([advantages_from_optimizer[i] for i in valid_indices], dtype=torch.float).cuda()
    updated_memory_dataframe = update_molecule_memory(
        current_memory_dataframe, smiles_generated, advantages_from_optimizer, score_dicts_evaluated, seqs_list_for_optimizer, nums_list_for_optimizer, training_arguments.memory_size
    )
    er_seqs, er_nums, er_valid_mask, er_smiles, er_scores, er_score_dicts = prepare_experience_replay(
        mo_optimizer_instance, generated_encoded_sequences.shape[1], training_arguments.replay
    )
    if er_seqs:
        combined_encoded_sequences = torch.cat((valid_encoded_seqs, torch.stack(er_seqs)), dim=0)
        combined_numbers_sequences = torch.cat((valid_numbers_seqs, torch.stack(er_nums)), dim=0)
        combined_advantages = torch.cat((valid_advantages, torch.tensor(er_scores, dtype=torch.float).cuda()), dim=0)
    else:
        combined_encoded_sequences = valid_encoded_seqs
        combined_numbers_sequences = valid_numbers_seqs
        combined_advantages = valid_advantages
    total_loss, policy_loss, kl_penalty, kl_coefficient, entropy, kl_divergence_mean = compute_reinforce_loss(
        agent_model, prior_model, combined_encoded_sequences, combined_numbers_sequences, combined_advantages, entropy_coefficient_value
    )  
    _, grad_norm = optimize_agent_parameters(agent_model, optimizer_instance, total_loss)
    
    return (
        agent_model, 
        mo_optimizer_instance,
        updated_memory_dataframe, 
        total_loss.item(), 
        policy_loss.item(), 
        kl_penalty.item(), 
        kl_coefficient, 
        entropy.item(), 
        entropy_coefficient_value, 
        temperature_value, 
        kl_divergence_mean, 
        advantages.mean().item(),
        advantages.std().item(),
        sum(valid_mask_generated) / training_arguments.batch_size,
        smiles_generated,
        score_dicts_evaluated,
        advantages.cpu().numpy(),
        grad_norm
    )
def run_datapoint_generation(datapoint_data, datapoint_index, prior_model, initial_agent_model, tokenizer_instance, vocabulary_instance, training_arguments):
    prot_file = os.path.join(training_arguments.data_path, 'test_set', datapoint_data['protein_filename'])
    data_tag = f"{datapoint_index}-{datapoint_data['protein_filename'].split('_rec')[0].replace('/', '-')}"   
    output_directory = os.path.join(training_arguments.mol_save_path, training_arguments.run_name + data_tag)
    os.makedirs(output_directory, exist_ok=True)
    prot_coords = datapoint_data['protein_pos'].numpy() - datapoint_data['ligand_center_of_mass'].numpy()
    tokens_initial, numbers_initial = tokenizer_instance.tokenize(protein=datapoint_data['protein_atom_name'], prot_coords=prot_coords)
    tokens_initial.append('<LIG_START>')
    numbers_initial.append(1)
    encoded_initial = vocabulary_instance.encode(tokens_initial)
    encoded_batch_initial = torch.tensor(encoded_initial, dtype=torch.long).unsqueeze(0).expand(training_arguments.batch_size, -1).to("cuda")
    numbers_batch_initial = torch.tensor(numbers_initial, dtype=torch.float).unsqueeze(0).expand(training_arguments.batch_size, -1).to("cuda")
    initial_memory = pd.DataFrame(columns=["smiles", "scores", "docking", "qed", "sa", "steric", "torsion", "energy", "bioactivity", "seqs", "numbers"])
    initial_mo_optimizer = MultiObjectiveOptimizer(batch_size=training_arguments.batch_size)
    optimizer_instance = optim.AdamW(initial_agent_model.parameters(), 
                                     lr=5e-6,
                                     betas=(0.9, 0.999), 
                                     weight_decay=0.01)
    current_agent_state = initial_agent_model
    current_mo_optimizer_state = initial_mo_optimizer
    current_memory_state = initial_memory
    valid_counts_first_25 = []
    total_valid_first_25 = 0
    for step_idx in range(training_arguments.n_steps):
        current_agent_state, current_mo_optimizer_state, current_memory_state, _, _, _, _, _, _, _, _, _, _, valid_ratio, _, _, _, _ = (
            process_training_step(
                step_idx, current_agent_state, prior_model, current_mo_optimizer_state, optimizer_instance, 
                encoded_batch_initial, numbers_batch_initial, 
                prot_file, datapoint_data['ligand_center_of_mass'].numpy(), 
                training_arguments, current_memory_state, tokenizer_instance, vocabulary_instance
            )
        )
        if step_idx < 25:
            valid_count = int(valid_ratio * training_arguments.batch_size)
            valid_counts_first_25.append(valid_count)
            total_valid_first_25 += valid_count
        if step_idx == 24:
            steps_with_valid = sum(1 for count in valid_counts_first_25 if count > 0)            
            print(f"\n{'='*80}")
            print(f"CHECKPOINT @ Step 25: Total valid={total_valid_first_25}, Steps with ≥1 valid={steps_with_valid}/25")
            if steps_with_valid < 10:
                print(f"⚠️  SKIPPING PROTEIN {datapoint_index}: Insufficient valid generation ({steps_with_valid}/25 steps)")
                print(f"{'='*80}\n")
                final_memory = current_memory_state
                pareto_frontier = calculate_pareto_frontier(final_memory, top_n_molecules=100) if len(final_memory) > 0 else pd.DataFrame()
                return final_memory, pareto_frontier
            else:
                print(f"✓ CONTINUING: Protein shows promise ({steps_with_valid}/25 steps with valid molecules)")
                print(f"{'='*80}\n")        
        if step_idx % 50 == 0:
            current_memory_state.to_csv(os.path.join(output_directory, f"step{step_idx}.csv"), index=False)
        if step_idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    final_memory = current_memory_state
    final_memory.to_csv(os.path.join(output_directory, "final.csv"), index=False)    
    pareto_frontier = calculate_pareto_frontier(final_memory, top_n_molecules=100)
    pareto_frontier.to_csv(os.path.join(output_directory, "pareto_frontier_top100.csv"), index=False)
    return final_memory, pareto_frontier
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='LigandGPT RL Fine-tuning for Molecule Generation')
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--data_path', type=str, default="./data",
                        help='Path to the data directory')
    parser.add_argument('--vocab_path', type=str, default="./ProtLigVoc.txt",
                        help='Path to the vocabulary file')
    parser.add_argument('--ckpt_load_path', type=str, default=None,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--mol_save_path', type=str, default="./generated_molecules/",
                        help='Directory to save generated molecules')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=12, required=False)
    parser.add_argument('--n_head', type=int, default=12, required=False)
    parser.add_argument('--n_embd', type=int, default=768, required=False)
    parser.add_argument('--n_steps', type=int, default=700)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--memory_size', type=int, default=500)
    parser.add_argument('--replay', type=int, default=8)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    tokenizer_instance = ProtLigTokenizer()
    vocabulary_instance = read_vocabulary(args.vocab_path)    
    prior_model_instance = LigandGPT(vocab_size=vocabulary_instance.__len__(), 
                                     d_model=args.n_embd, nhead=args.n_head, num_layers=args.n_layer, 
                                     dim_feedforward=4 * args.n_embd, context_length=args.max_length).to("cuda")
    prior_model_instance.load_state_dict(torch.load(args.ckpt_load_path), strict=True)
    for param in prior_model_instance.parameters():
        param.requires_grad = False
    prior_model_instance.eval()
    txn_data, keys_data = get_lmdb_data(os.path.join(args.data_path, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"))
    ids_data = torch.load(os.path.join(args.data_path, "crossdocked_pocket10_pose_split.pt"))['test']
    def process_all_datapoints(datapoint_indices):
        results_per_datapoint = []
        for datapoint_idx in datapoint_indices:
            agent_model_instance = LigandGPT(vocab_size=vocabulary_instance.__len__(), 
                                             d_model=args.n_embd, nhead=args.n_head, num_layers=args.n_layer, 
                                             dim_feedforward=4 * args.n_embd, context_length=args.max_length).to("cuda")
            agent_model_instance.load_state_dict(torch.load(args.ckpt_load_path), strict=True)
            agent_model_instance.train()            
            datapoint_pickled = txn_data.get(keys_data[ids_data[datapoint_idx]])
            current_datapoint = pickle.loads(datapoint_pickled)            
            final_memory_df, final_pareto_df = run_datapoint_generation(
                current_datapoint, datapoint_idx, prior_model_instance, agent_model_instance, tokenizer_instance, vocabulary_instance, args
            )
            results_per_datapoint.append((final_memory_df, final_pareto_df))
            del agent_model_instance
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()            
            print(f"\n{'='*80}")
            print(f"Completed protein {datapoint_idx}, GPU memory cleared for next target")
            print(f"{'='*80}\n")            
        return results_per_datapoint
    process_all_datapoints(range(0, 100))
if __name__ == "__main__":
    main()