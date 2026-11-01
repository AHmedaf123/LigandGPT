"""Seconds per RL step on target 0 (batch 32/48/64): sample (KV cache) -> score+dock (parallel) -> policy-update cost.
Run: .venv/bin/python bench_step.py"""
import json
import time

import numpy as np
import torch

import decoding as dec
import diagnose_generation as dg
import generation_finetuning as gf
import scoring_function as sf
from data_utils import receptor_paths
from model import LigandGPT
from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary

voc, tk = read_vocabulary("./ProtLigVoc.txt"), ProtLigTokenizer()
mk = lambda: LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()  # noqa: E731
model = mk()
model.load_state_dict(torch.load("final.pt"))
model.eval()
ids, nums, _ = dg.load_prompt(0, voc, tk)
L = ids.shape[1]
tm = json.load(open("data/target_map.json"))["0"]
rec = receptor_paths(tm["protein_filename"], "data")[1]
sf.initialize_scoring_system()
out = {"prompt_tokens": L, "load_average": open("/proc/loadavg").read().split()[:3]}


def sync():
    torch.cuda.synchronize()
    return time.time()


rows = {}
for n in (32, 48, 64):
    torch.manual_seed(n)
    t0 = sync()
    gen, _ = dec.sample_smiles(model, voc, ids, nums, n, structured=True, valence=True, top_k=10, max_new_tokens=200)
    t_sample = sync() - t0
    toks = dg.rows_to_tokens(gen, voc, voc["<LIG_END>"])
    smiles = ["".join(t) for t, _ in toks]
    n_valid = sum(dg.classify(t, term)[1] for t, term in toks)
    sf.COUNTERS.clear()
    t0 = time.time()
    sf.calculate_scores(smiles, receptor_file=rec, box_center=tm["center"])
    t_score = time.time() - t0
    rows[n] = dict(sampling_s=t_sample, scoring_incl_docking_s=t_score, valid=n_valid, counters=dict(sf.COUNTERS))
    print(n, rows[n], flush=True)
out["per_batch"] = rows

# policy-update cost: forward+backward of the real loss on micro-batches of full-length sequences (prompt + 100 generated tokens)
agent, prior = mk(), mk()
agent.load_state_dict(torch.load("final.pt"))
prior.load_state_dict(torch.load("final.pt"))
prior.eval()
for p in prior.parameters():
    p.requires_grad_(False)
agent.train()
upd = {}
for mb in (2, 4):
    try:
        x = torch.cat([ids.expand(mb, -1), torch.randint(1, 60, (mb, 100)).cuda()], 1)
        nm = torch.ones_like(x, dtype=torch.float)
        torch.cuda.reset_peak_memory_stats()
        t0 = sync()
        loss = gf.compute_reinforce_loss(agent, prior, x, nm, torch.ones(mb).cuda(), 0.1)[0]
        loss.backward()
        upd[mb] = dict(seconds=sync() - t0, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        agent.zero_grad(set_to_none=True)
        del loss
    except torch.OutOfMemoryError:
        upd[mb] = "OOM"
        torch.cuda.empty_cache()
    print("update micro-batch", mb, upd[mb], flush=True)
out["update_microbatch"] = upd
json.dump(out, open("runs/baseline/step_bench.json", "w"), indent=1)
