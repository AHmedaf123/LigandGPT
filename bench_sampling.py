"""KV-cache check + timing for the structured decoder on target 0. Run: .venv/bin/python bench_sampling.py"""
import cProfile
import json
import pstats
import time

import torch

import decoding as dec
import diagnose_generation as dg
from model import LigandGPT
from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary

voc, tk = read_vocabulary("./ProtLigVoc.txt"), ProtLigTokenizer()
model = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
model.load_state_dict(torch.load("final.pt"))
model.eval()
ids, nums, _ = dg.load_prompt(0, voc, tk)
L = ids.shape[1]
KW = dict(structured=True, valence=True, top_k=10, temperature=1.0)


def sample(n, seed, cache, max_new=200):
    torch.manual_seed(seed)
    return dec.sample_smiles(model, voc, ids, nums, n, max_new_tokens=max_new, use_cache=cache, **KW)[0]


def sync_time(f):
    torch.cuda.synchronize()
    t0 = time.time()
    r = f()
    torch.cuda.synchronize()
    return r, time.time() - t0


# 1. cached vs uncached logits on 8 sequences: teacher-force the sampled tokens through both paths
rows = sample(8, 123, cache=False)
T = max(len(r) for r in rows)
full = torch.cat([ids.expand(8, -1), torch.tensor([r + [0] * (T - len(r)) for r in rows]).cuda()], 1)
with torch.no_grad():
    ref = model(full, torch.ones_like(full, dtype=torch.float))[0][:, L - 1:L - 1 + T]          # logits that predict generated token t
    kvp, lg, _ = dec.prefill(model, ids, torch.ones_like(ids, dtype=torch.float))
    kvn, got = [None] * 12, [lg.expand(8, -1)]
    for t in range(T - 1):
        lg, _ = dec.decode_step(model, full[:, L + t:L + t + 1], torch.ones(8, 1).cuda(), L + t, kvp, kvn)
        got.append(lg)
diff = (torch.stack(got, 1) - ref).abs().max().item()
print(f"1. max |cached - uncached| logit over 8 sequences x {T} positions: {diff:.2e}  ({'OK' if diff < 1e-3 else 'FAIL'}, limit 1e-3)")

# 2. same seeds as baseline.py (target 0: 8 batches of 16, seeds 2000+b): validity and identical molecules vs the saved uncached run
samples = json.load(open("runs/baseline/samples.json"))[0]
old = [s for s in samples if s["target"] == 0]
new = []
for b in range(8):
    for toks, term in dg.rows_to_tokens(sample(16, 2000 + b, cache=True), voc, voc["<LIG_END>"]):
        cat, ok, _ = dg.classify(toks, term)
        new.append(dict(smiles="".join(toks), valid=ok))
same = sum(a["smiles"] == b["smiles"] for a, b in zip(old, new))
print(f"2. target 0, 128 samples, seeds 2000-2007: valid uncached {sum(s['valid'] for s in old)}  cached {sum(s['valid'] for s in new)} | identical SMILES {same}/128")

# 3. timing, one call per size
for n in (16, 32, 48, 64):
    _, t = sync_time(lambda: sample(n, 7, cache=True))
    print(f"3. sample {n:2d} molecules: {t:5.1f} s  ({t / n:.2f} s/molecule)")
_, t = sync_time(lambda: sample(16, 7, cache=False))
print(f"   (uncached, 16 molecules: {t:.1f} s)")

# 4. where does the time go at n=64?
pr = cProfile.Profile()
pr.enable()
sample(64, 7, cache=True)
torch.cuda.synchronize()
pr.disable()
st = pstats.Stats(pr)
for name in ("build_mask", "allowed", "advance", "decode_step", "multinomial", "topk", "masked_fill"):
    rows_ = [(k, v) for k, v in st.stats.items() if k[2].endswith(name) or k[2].endswith(f"'{name}'") or name in k[2]]
    if rows_:
        k, v = max(rows_, key=lambda kv: kv[1][3])
        print(f"4. {name:12s} cumulative {v[3]:6.2f} s  calls {v[1]}")
