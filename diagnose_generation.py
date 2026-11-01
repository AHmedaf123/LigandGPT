"""Free-generation diagnosis for the pretrained LigandGPT prior.

For targets 0 and 3 (pocket + coordinate prompt from the restored tokenizer), draw N samples per decoder setting with
identical seeds, classify every sample, and write results/generation_diagnosis/{samples,summary}.csv and summary.md.

Decoders:
  plain_*       original 3DMolFormer sampler: temperature 1.0, optional top-k, no masks, no biases
  current_*     training_utils.predict_smiles as it is in the repo (atom biases, token masks, T halved for 30 steps)
  struct_*      decoding.sample_smiles: grammar masks only (syntax) or grammar + valence, no biases

Validity (headline, "strict"): the sample ended with <LIG_END>, RDKit parses AND sanitizes the raw SMILES (no SELFIES
repair) and it has >= 4 heavy atoms. "pipeline" = what generation_finetuning counts as valid (SELFIES round trip first).

Run:  .venv/bin/python diagnose_generation.py --targets 0 3 --n 64 --out results/generation_diagnosis
"""
import argparse
import collections
import csv
import os
import pickle
import sys
import time

import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import selfies as sf  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

import decoding as dec  # noqa: E402
from model import LigandGPT  # noqa: E402
from tokenizer import ProtLigTokenizer  # noqa: E402
from vocabulary import read_vocabulary  # noqa: E402

RDLogger.DisableLog("rdApp.*")
DATA = "/home/coder/project/data"
FORBIDDEN = set(dec.NON_SMILES) | set(dec.NON_ORGANIC)
SETTINGS = [  # name, kind, kwargs
    ("plain_k10", "plain", dict(top_k=10, temperature=1.0)),
    ("plain_full", "plain", dict(top_k=0, temperature=1.0)),
    ("current_k10", "current", dict(top_k=10, temperature=1.0)),
    ("current_rl_step0", "current", dict(top_k=70, temperature=0.9)),
    ("struct_syntax_k10", "struct", dict(top_k=10, temperature=1.0, valence=False)),
    ("struct_syntax_full", "struct", dict(top_k=0, temperature=1.0, valence=False)),
    ("struct_valence_k10", "struct", dict(top_k=10, temperature=1.0, valence=True)),
    ("struct_valence_full", "struct", dict(top_k=0, temperature=1.0, valence=True)),
]


# ------------------------------------------------------------------------------------------------ classification
def violation_type(tok):
    if tok in ("[x]", "[y]", "[z]"):
        return "coordinate_token"
    if tok in dec.NON_ORGANIC:
        return "non_organic_token"
    if tok in (".", "/", "%10", "%11"):
        return "disallowed_symbol"
    if tok == "<PAD>":
        return "pad_inside_smiles"
    return "protein_or_special_token"


def syntax_reason(tokens):
    if not tokens:
        return "empty"
    atoms_like = lambda t: t not in dec.BOND_ORDER and t not in ("(", ")") and t not in dec.RING_DIGITS  # noqa: E731
    if not atoms_like(tokens[0]):
        return "starts_with_non_atom"
    depth = 0
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if t == "(":
            depth += 1
            if nxt == ")":
                return "empty_branch"
            if nxt in dec.RING_DIGITS:
                return "ring_digit_after_open_paren"
        elif t == ")":
            depth -= 1
            if depth < 0:
                return "unmatched_close_paren"
            if nxt in dec.RING_DIGITS:
                return "ring_digit_after_close_paren"
        elif t in dec.BOND_ORDER and (nxt is None or nxt in dec.BOND_ORDER or nxt in ("(", ")")):
            return "dangling_or_double_bond"
    if depth > 0:
        return "unclosed_branch"
    counts = collections.Counter(t for t in tokens if t in dec.RING_DIGITS)
    if any(c % 2 for c in counts.values()):
        return "unclosed_ring_digit"
    return "other_syntax"


def classify(tokens, terminated):
    """Return (category, strict_valid, n_heavy)."""
    bad = [t for t in tokens if t in FORBIDDEN]
    if bad:
        return "type_violation:" + violation_type(bad[0]), False, 0
    if not terminated:
        return "no_termination", False, 0
    smi = "".join(tokens)
    m = Chem.MolFromSmiles(smi, sanitize=False) if smi else None
    if m is None:
        return "syntax:" + syntax_reason(tokens), False, 0
    n_heavy = m.GetNumAtoms()
    if n_heavy < 4:
        return "too_small(<4 atoms)", False, n_heavy
    flag = Chem.SanitizeMol(m, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        name = str(flag)
        return {"SANITIZE_KEKULIZE": "rdkit:kekulize(aromatic ring invalid)", "SANITIZE_PROPERTIES": "rdkit:valence"}.get(name, "rdkit:" + name), False, n_heavy
    return "valid", True, n_heavy


def pipeline_valid(tokens):
    smi = "".join(tokens)
    if not smi:
        return False
    try:
        m = Chem.MolFromSmiles(sf.decoder(sf.encoder(smi)))
    except Exception:
        m = Chem.MolFromSmiles(smi)
    return m is not None and m.GetNumAtoms() >= 4


# ------------------------------------------------------------------------------------------------ generation
def load_prompt(target, voc, tk):
    import lmdb
    split = torch.load(f"{DATA}/crossdocked_pocket10_pose_split.pt", weights_only=False)["test"]
    env = lmdb.open(f"{DATA}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb", subdir=False, readonly=True,
                    lock=False, readahead=False, meminit=False, max_readers=256)
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        dp = pickle.loads(txn.get(keys[split[target]]))
    coords = dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy()
    toks, nums = tk.tokenize(protein=dp["protein_atom_name"], prot_coords=coords)
    toks.append("<LIG_START>")
    nums.append(1)
    ids = torch.tensor(voc.encode(toks), dtype=torch.long).unsqueeze(0).cuda()
    return ids, torch.tensor(nums, dtype=torch.float).unsqueeze(0).cuda(), dp["protein_filename"]


def rows_to_tokens(rows, voc, end_id):
    out = []
    for r in rows:
        toks, terminated = [], False
        for i in r:
            if i == end_id:
                terminated = True
                break
            if i == 0:                      # padding after a stop
                break
            toks.append(voc.decode([i])[0])
        out.append((toks, terminated))
    return out


def generate(kind, kw, model, voc, ids, nums, n, max_new, seed):
    torch.manual_seed(seed)
    end_id = voc[dec.END]
    if kind == "current":
        import training_utils as tu
        out_ids, _ = tu.predict_smiles(model, ids.expand(n, -1).clone(), nums.expand(n, -1).clone(), max_len=max_new,
                                       temperature=kw["temperature"], top_k=kw["top_k"])
        rows = out_ids[:, ids.shape[1]:].tolist()
    else:
        rows, _ = dec.sample_smiles(model, voc, ids, nums, n, structured=(kind == "struct"), temperature=kw["temperature"],
                                    top_k=kw["top_k"], max_new_tokens=max_new, valence=kw.get("valence", False))
    return rows_to_tokens(rows, voc, end_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--n", type=int, default=64, help="samples per setting and target")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--settings", nargs="*", default=None)
    ap.add_argument("--out", default="results/generation_diagnosis")
    ap.add_argument("--ckpt", default="final.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    voc, tk = read_vocabulary("./ProtLigVoc.txt"), ProtLigTokenizer()
    model = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    samples = []
    t0 = time.time()
    for target in args.targets:
        ids, nums, pname = load_prompt(target, voc, tk)
        print(f"target {target} ({pname}) prompt tokens {ids.shape[1]}, budget {min(args.max_new, 2048 - ids.shape[1])} new tokens", flush=True)
        for name, kind, kw in SETTINGS:
            if args.settings and name not in args.settings:
                continue
            budget = min(args.max_new, 2048 - ids.shape[1])
            for b in range(args.n // args.batch):
                for k, (toks, term) in enumerate(generate(kind, kw, model, voc, ids, nums, args.batch, budget, args.seed + b)):
                    cat, ok, nh = classify(toks, term)
                    samples.append(dict(target=target, setting=name, batch=b, idx=k, terminated=int(term), n_tokens=len(toks),
                                        category=cat, strict_valid=int(ok), pipeline_valid=int(pipeline_valid(toks)),
                                        n_heavy=nh, smiles="".join(toks), tokens=" ".join(toks)))
            rows = [s for s in samples if s["target"] == target and s["setting"] == name]
            print(f"  [{time.time() - t0:6.0f}s] {name:20s} strict {sum(s['strict_valid'] for s in rows):3d}/{len(rows)}  "
                  f"pipeline {sum(s['pipeline_valid'] for s in rows):3d}/{len(rows)}", flush=True)
            with open(os.path.join(args.out, "samples.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
                w.writeheader()
                w.writerows(samples)
    write_summary(samples, args.out)


def write_summary(samples, out):
    names = [s[0] for s in SETTINGS if any(x["setting"] == s[0] for x in samples)]
    targets = sorted({s["target"] for s in samples})
    lines = ["# Free-generation diagnosis", "",
             "strict = terminated + RDKit parse + sanitize + >= 4 heavy atoms (raw SMILES, no SELFIES repair); "
             "pipeline = SELFIES round trip as in generation_finetuning.validate_and_standardize_smiles.", ""]
    hdr = "| setting | " + " | ".join(f"T{t} strict | T{t} pipeline" for t in targets) + " | both strict | unique valid | mean len |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    summ = []
    for n in names:
        cells, tot, ok, tp = [], 0, 0, 0
        for t in targets:
            r = [s for s in samples if s["setting"] == n and s["target"] == t]
            cells += [f"{sum(s['strict_valid'] for s in r)}/{len(r)}", f"{sum(s['pipeline_valid'] for s in r)}/{len(r)}"]
            tot += len(r)
            ok += sum(s["strict_valid"] for s in r)
        r = [s for s in samples if s["setting"] == n]
        uniq = len({s["smiles"] for s in r if s["strict_valid"]})
        ml = sum(s["n_tokens"] for s in r) / max(len(r), 1)
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {ok / max(tot, 1):.2f} | {uniq} | {ml:.0f} |")
        summ.append(dict(setting=n, n=tot, strict_valid=ok, strict_rate=ok / max(tot, 1), unique_valid=uniq, mean_len=ml))
    lines += ["", "## Failure breakdown (count of samples, both targets pooled)", ""]
    cats = sorted({s["category"] for s in samples})
    lines += ["| category | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(str(sum(1 for s in samples if s["setting"] == n and s["category"] == c)) for n in names) + " |")
    lines += ["", "## Atom token usage (share of atom tokens)", "", "| setting | aliphatic C/N/O | aromatic c/n/o | halogens | bracket atoms |", "|---|---|---|---|---|"]
    for n in names:
        toks = [t for s in samples if s["setting"] == n for t in s["tokens"].split()]
        atoms = [t for t in toks if t not in dec.BOND_ORDER and t not in ("(", ")") and t not in dec.RING_DIGITS and t not in FORBIDDEN]
        c = collections.Counter(atoms)
        tot = max(len(atoms), 1)
        lines.append(f"| {n} | {sum(c[x] for x in 'CNO') / tot:.2f} | {sum(c[x] for x in 'cno') / tot:.2f} | "
                     f"{sum(c[x] for x in ('F', 'Cl', 'Br', 'I')) / tot:.2f} | {sum(v for k, v in c.items() if k.startswith('[')) / tot:.2f} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
