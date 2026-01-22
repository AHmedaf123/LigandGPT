import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from vocabulary import read_vocabulary
from tokenizer import ProtLigTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_current_dir = os.path.dirname(os.path.abspath(__file__))
vocab = read_vocabulary(os.path.join(_current_dir, "ProtLigVoc.txt"))
num_token_id = torch.tensor([vocab.__getitem__('[x]'), 
                             vocab.__getitem__('[y]'), 
                             vocab.__getitem__('[z]')]).to(device)
lig_smiles_start_id = torch.tensor(vocab.__getitem__('<LIG_START>')).to(device)
lig_smiles_end_id = torch.tensor(vocab.__getitem__('<LIG_END>')).to(device)
lig_coord_start_id = torch.tensor(vocab.__getitem__('<LIG_COORDS_START>')).to(device)
lig_coord_end_id = torch.tensor(vocab.__getitem__('<LIG_COORDS_END>')).to(device)
tokenizer = ProtLigTokenizer()
common_atoms = ['C', 'c', 'N', 'n', 'O', 'o', 'F', '(', ')', '=', '-', '1', '2', '3', '[C]', '[N]', '[O]', '<LIG_END>']
common_atom_ids = torch.tensor([vocab[t] for t in common_atoms if t in vocab]).to(device)
rare_atoms = ['S', 's', 'P', '[S]', '[P]', '[Se]', '[S+]', '[Se+]', '[P+]']
rare_atom_ids = torch.tensor([vocab[t] for t in rare_atoms if t in vocab]).to(device)
isotope_tokens = ['[17O]', '[18F]', '[2H]', '[As]', '[Fe]', '[K]', '[K+]', '[Li+]', '[Mg+2]', '[Ca+2]', '[Na]', '[Na+]', '[Sn]', '[V]']
isotope_token_ids = torch.tensor([vocab[t] for t in isotope_tokens if t in vocab]).to(device)
ligand_tokens = ['C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'F', 'Cl', 'Br', 'I', 
                 '(', ')', '=', '#', '-', '+', '[', ']',
                 '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '[C]', '[N]', '[O]', '[S]', '[P]', '[Se]', '[B]', '[Si]',
                 '[C@H]', '[C@@H]', '[N+]', '[O-]', '[S+]', '[n+]', '<LIG_END>']
ligand_token_ids = torch.tensor([vocab[t] for t in ligand_tokens if t in vocab]).to(device)
protein_tokens = ['CA', 'N', 'C', 'O', '<PROT_END>', '<PROT_COORDS_START>', '<PROT_COORDS_END>']
protein_token_ids = torch.tensor([vocab[t] for t in protein_tokens if t in vocab]).to(device)
invalid_tokens = ['<PAD>', '<PROT_START>', '[V]', '[x]', '[y]', '[z]', '#', '%10', '%11', '.', '/']
invalid_token_ids = torch.tensor([vocab[t] for t in invalid_tokens if t in vocab]).to(device)
no_start_tokens = [')', '=', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', ']']
no_start_token_ids = torch.tensor([vocab[t] for t in no_start_tokens if t in vocab]).to(device)
def get_lr(it, total_it, learning_rate, warmup):
    warmup_iters = warmup * total_it
    if it < warmup_iters:      
        lr_mult = it / warmup_iters
    else:      
        decay_ratio  = (it - warmup_iters) / (total_it - warmup_iters)
        lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay_ratio)))
    return learning_rate * lr_mult
def loss_joint(logits, nums, y, y_num, loss_num_w=1.0):
    num_mask = torch.isin(y, num_token_id)
    loss_lm = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                              ignore_index=0, reduction="mean")
    loss_num = F.mse_loss(nums[num_mask], y_num[num_mask].view(-1, 1),
                          reduction="mean")
    loss = loss_lm + loss_num_w * loss_num
    return loss, loss_lm, loss_num
def loss_docking(nums, y, y_num):
    N, L = y.shape
    lig_coord_mask = torch.zeros_like(y, dtype=torch.bool)
    for i in range(N):
        start_pos = (y[i] == lig_coord_start_id).nonzero(as_tuple=True)[0].item()
        end_pos = (y[i] == lig_coord_end_id).nonzero(as_tuple=True)[0].item()
        range_mask = torch.zeros(L, dtype=torch.bool).to(device)
        range_mask[start_pos + 1:end_pos] = True
        lig_coord_mask[i] = range_mask & torch.isin(y[i], num_token_id)
    loss = F.mse_loss(nums[lig_coord_mask], y_num[lig_coord_mask].view(-1, 1), 
                      reduction="mean")
    return loss
def likelihood(model, seqs, numbers):
    nll_loss = nn.NLLLoss(reduction="none")
    seqs = seqs.cuda()
    numbers = numbers.cuda()
    logits, _ = model(seqs[:, :-1], numbers[:, :-1])
    log_probs = logits.log_softmax(dim=2)
    return nll_loss(log_probs.transpose(1, 2), seqs[:, 1:]).sum(dim=1)
def predict(model, token_sequences, number_sequences, max_len=2048, temperature=1.0, top_k=10, device='cuda'):
    model.eval()
    token_sequences = token_sequences.to(device)
    number_sequences = number_sequences.to(device)
    N, seq_len = token_sequences.size()
    output_tokens = token_sequences.clone()
    output_numbers = number_sequences.clone()
    for step in range(seq_len, max_len):
        with torch.no_grad():
            logits, num_preds = model(output_tokens, output_numbers)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tokens = top_k_indices.gather(-1, next_token_indices.unsqueeze(-1)).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        output_tokens = torch.cat([output_tokens, next_tokens.unsqueeze(1)], dim=1)
        next_numbers = num_preds[:, -1, :].squeeze(-1)
        output_numbers = torch.cat([output_numbers, next_numbers.unsqueeze(1)], dim=1)
        for i in range(N):
            if lig_smiles_end_id in output_tokens[i]:
                start_idx = (output_tokens[i] == lig_smiles_start_id).nonzero(as_tuple=True)[0].item()
                end_idx = (output_tokens[i] == lig_smiles_end_id).nonzero(as_tuple=True)[0].item()
                ligand_smiles_codes = output_tokens[i, start_idx:end_idx + 1]
                ligand_smiles_tokens = vocab.decode(ligand_smiles_codes.cpu().numpy())
                ligand_smiles = tokenizer.untokenize(ligand_smiles_tokens)['LigSmiles']                
                try:
                    mol = Chem.MolFromSmiles(ligand_smiles)
                    num_atoms = mol.GetNumAtoms()
                except:
                    num_atoms = 0
                curr_idx = len(output_tokens[i]) - 1
                if curr_idx == end_idx + 1:
                    output_tokens[i, -1] = lig_coord_start_id
                coord_start_idx = end_idx + 2
                if curr_idx >= coord_start_idx and curr_idx < coord_start_idx + 3 * num_atoms:
                    output_tokens[i, -1] = num_token_id[(curr_idx - coord_start_idx) % 3]
                elif curr_idx == coord_start_idx + 3 * num_atoms:
                    output_tokens[i, -1] = lig_coord_end_id
                elif curr_idx > coord_start_idx + 3 * num_atoms:
                    output_tokens[i, -1] = 0
            if output_tokens[i, -1] not in num_token_id:
                output_numbers[i, -1] = 1.0     
        if (output_tokens[:, -1] == 0).all():
            break
    return output_tokens, output_numbers
def predict_smiles(model, token_sequences, number_sequences, max_len=2048, temperature=1.0, top_k=10, device='cuda', debug=False):
    model.eval()
    token_sequences = token_sequences.to(device)
    number_sequences = number_sequences.to(device)    
    N, input_len = token_sequences.size()
    output_tokens = token_sequences.clone()
    output_numbers = number_sequences.clone()
    max_context_length = 2048
    max_new_tokens = min(max_len, max_context_length - input_len)
    if debug:
        print(f"\n=== predict_smiles DEBUG ===")
        print(f"Input length: {input_len}, Max new tokens: {max_new_tokens}")
        print(f"Last 10 input tokens: {vocab.decode(token_sequences[0, -10:].cpu().numpy())}")
    from tqdm import tqdm
    for step in tqdm(range(max_new_tokens), leave=False, disable=debug):        
        with torch.no_grad():
            logits, num_preds = model(output_tokens, output_numbers)
        for i in range(N):
            if lig_smiles_start_id in output_tokens[i] and lig_smiles_end_id not in output_tokens[i]:
                logits[i, -1, invalid_token_ids] = -1e10
                logits[i, -1, protein_token_ids] = -1e10
                logits[i, -1, isotope_token_ids] = -1e10
                if step == 0:
                    logits[i, -1, no_start_token_ids] = -1e10
                if step < 5:
                    logits[i, -1, common_atom_ids] += 20.0
                    logits[i, -1, rare_atom_ids] = -1e10
                elif step < 20:
                    logits[i, -1, common_atom_ids] += 10.0
                    logits[i, -1, rare_atom_ids] -= 15.0
                else:
                    logits[i, -1, ligand_token_ids] += 8.0
                    logits[i, -1, rare_atom_ids] -= 8.0
        for i in range(N):
            if lig_smiles_start_id in output_tokens[i] and lig_smiles_end_id not in output_tokens[i]:
                start_idx = (output_tokens[i] == lig_smiles_start_id).nonzero(as_tuple=True)[0][0].item()
                current_tokens = vocab.decode(output_tokens[i, start_idx+1:].cpu().numpy())
                current_smiles = ''.join([t for t in current_tokens if t not in ['[x]', '[y]', '[z]', '<PROT_END>']])
                open_paren = current_smiles.count('(')
                close_paren = current_smiles.count(')')
                if close_paren >= open_paren:
                    close_paren_id = vocab[')']
                    logits[i, -1, close_paren_id] = -1e10
        temp = temperature if step > 30 else temperature * 0.5
        logits = logits[:, -1, :] / temp
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tokens = top_k_indices.gather(-1, next_token_indices.unsqueeze(-1)).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)     
        if debug and step < 30:
            print(f"Step {step}: Generated token ID={next_tokens[0].item()}, Token={vocab.decode([next_tokens[0].item()])[0]}")
            if step == 29:
                print(f"First 30 generated tokens: {vocab.decode(output_tokens[0, input_len:].cpu().numpy())}")     
        output_tokens = torch.cat([output_tokens, next_tokens.unsqueeze(1)], dim=1)
        next_numbers = num_preds[:, -1, :].squeeze(-1)
        output_numbers = torch.cat([output_numbers, next_numbers.unsqueeze(1)], dim=1)
        for i in range(N):
            if lig_smiles_end_id in output_tokens[i]:
                end_idx = (output_tokens[i] == lig_smiles_end_id).nonzero(as_tuple=True)[0][0].item()
                curr_idx = len(output_tokens[i]) - 1
                if curr_idx > end_idx:
                    output_tokens[i, -1] = 0
            if output_tokens[i, -1] == lig_coord_end_id:
                output_tokens[i, -1] = 0
            if output_tokens[i, -1] not in num_token_id:
                output_numbers[i, -1] = 1.0        
        if (output_tokens[:, -1] == 0).all():
            break
    return output_tokens, output_numbers