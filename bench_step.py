"""Seconds per RL step on target 0 (batch 32/48/64): sample (KV cache) -> score+dock (parallel). (The update-cost block was removed when
the RL loss was rebuilt; its measurement is kept in runs/baseline/step_bench.json.)
Run: .venv/bin/python bench_step.py"""
import json
import time

import numpy as np
import torch

import decoding as dec
import diagnose_generation as dg
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

print(json.dumps(out))
