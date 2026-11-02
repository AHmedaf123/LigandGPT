"""Size-bias audit (offline, no RL run, no reward or gate is changed).
Question: do the reward terms and gates push the policy toward small molecules (the probes shrank valid molecules from ~30 to ~20 heavy atoms)?
collect : molecule set = 93 valid baseline molecules + 256 fresh prior samples (target 0, structured decoder, new seed; SMILES were not saved per
          step, only the 100-molecule Pareto archive of each probe) + the archives of the 3 probes (curriculum, fixed avg, fixed literal).
          Fresh molecules are scored like the RL loop (UFF/torsion/steric/QED/SA, GNN, QuickVina with the SMILES-derived seed).
analyse : Spearman of every normalised objective reward with heavy atoms, gate pass rate per size tercile, validity vs token length,
          docking / ligand efficiency vs heavy atoms, and what each candidate fix (A, B, C) would change. Nothing is implemented.
Run: .venv/bin/python size_audit.py [collect|analyse]     Writes results/size_audit/{molecules.csv,samples.csv,audit.md,audit.json}"""
import json
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import linregress, spearmanr

import baseline as bl                      # puts the repo on sys.path, chdirs to it; TMAP, DATA
import diagnose_generation as dg
import mo_mpo
import scoring_function as sf

warnings.filterwarnings("ignore")
OUT = "results/size_audit"
OBJ7 = ["docking", "bioactivity", "qed", "sa", "steric", "torsion", "energy"]
OBJ6 = ["docking", "bioactivity", "qed", "sa", "torsion", "energy"]
PROBES = ["probe_curriculum", "probe_fixed_avg", "probe_fixed_literal"]
OLD_STAGE2 = dict(docking=.30, bioactivity=.30, energy=.20, sa=.08, qed=.06, steric=.04, torsion=.02)     # weights used in the three probes
FRESH_SEED, N_FRESH, BATCH = 5000, 256, 16                                                              # baseline used seeds 2000-2007, 128 per target


# ---------------------------------------------------------------------------------------------------------------- collect
def sample_fresh():
    import torch
    from model import LigandGPT
    from tokenizer import ProtLigTokenizer
    from vocabulary import read_vocabulary
    import generation_finetuning as gf
    path = f"{OUT}/fresh_samples.json"
    if os.path.exists(path):
        return json.load(open(path))
    voc, tk = read_vocabulary("./ProtLigVoc.txt"), ProtLigTokenizer()
    gf.reserve_gpu(5)
    model = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
    model.load_state_dict(torch.load("final.pt"))
    model.eval()
    ids, nums, _ = dg.load_prompt(0, voc, tk)
    budget = min(200, 2048 - ids.shape[1])
    out = []
    for b in range(N_FRESH // BATCH):
        for toks, term in dg.generate("struct", dict(top_k=10, temperature=1.0, valence=True), model, voc, ids, nums, BATCH, budget, FRESH_SEED + b):
            cat, ok, _ = dg.classify(toks, term)
            out.append(dict(target=0, smiles="".join(toks), n_tokens=len(toks), category=cat, valid=ok))
        print(f"sampled {len(out)}/{N_FRESH}, valid {sum(s['valid'] for s in out)}", flush=True)
    json.dump(out, open(path, "w"))
    return out


def score_fresh(valid):
    oracles, gnn, _ = sf.initialize_scoring_system()
    for s in valid:
        try:
            md = sf.prepare_molecule_with_conformer(s["smiles"])
        except Exception:                      # RDKit raises in UFF for atom types without parameters: a 3D-build failure, as in the RL loop
            md = None
        p = sf.evaluate_property_scores(md, oracles["qed"], oracles["sa"]) if md is not None else None
        s.update(qed=p and p.qed, sa=p and p.synthetic_accessibility, steric=p and p.steric_clash, torsion=p and p.torsion_strain, energy=p and p.internal_energy)
    for s, b in zip(valid, sf.predict_bioactivity_for_batch([s["smiles"] for s in valid], gnn)):
        s["bioactivity"] = b
    m = bl.TMAP["0"]
    rec = bl.receptor_paths(m["protein_filename"], bl.DATA)[1]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(sf.cpu_quota()) as ex:
        def dock(smi):
            try:
                return sf.perform_qvina_docking(smi, rec, m["center"], td)
            except Exception:
                return None
        for s, d in zip(valid, ex.map(dock, [s["smiles"] for s in valid])):
            s["docking"] = d
    print(f"docked {len(valid)} fresh molecules in {time.time() - t0:.0f}s", flush=True)


def collect():
    os.makedirs(OUT, exist_ok=True)
    base = json.load(open("runs/baseline/baseline.json"))["molecules"]
    base_samples = json.load(open("runs/baseline/samples.json"))[0]
    fresh = sample_fresh()
    fresh_valid = [dict(s, src="fresh") for s in fresh if s["valid"]]
    score_fresh(fresh_valid)
    rows = [dict(m, src="baseline") for m in base if m["valid"]] + fresh_valid
    for p in PROBES:
        rows += [dict(a, src=p, target=0) for a in json.load(open(f"runs/{p}/target0/archive.json"))]
    df = pd.DataFrame(rows)[["src", "target", "smiles"] + OBJ7]
    df["heavy"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in df.smiles]
    df["canon"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df.smiles]
    df.to_csv(f"{OUT}/molecules.csv", index=False)
    pd.DataFrame([dict(s, src="baseline") for s in base_samples] + [dict(s, src="fresh") for s in fresh]).drop(columns=["valid"]).assign(
        valid=lambda d: d.category == "valid").to_csv(f"{OUT}/samples.csv", index=False)
    print(f"molecules {len(df)} (before de-duplication) by source: {df.src.value_counts().to_dict()}")


# ---------------------------------------------------------------------------------------------------------------- analyse
RNG = np.random.default_rng(0)


def rho(x, y, boot=1000):
    """Spearman with a bootstrap 95% CI over molecules -> (rho, lo, hi, n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan, len(x)
    bs = [spearmanr(x[i], y[i])[0] for i in (RNG.integers(0, len(x), len(x)) for _ in range(boot))]
    return spearmanr(x, y)[0], *np.nanpercentile(bs, [2.5, 97.5]), len(x)


def fmt_rho(t):
    return "n/a" if np.isnan(t[0]) else f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def slope10(x, y):
    """OLS slope of y on x per 10 units of x (x = heavy atoms)."""
    ok = np.isfinite(x) & np.isfinite(y)
    return 10 * linregress(x[ok], y[ok]).slope


def verdict(t, sign_name=("smaller", "larger")):
    """Rule: |rho| >= 0.15 with a CI that excludes 0 = a push; otherwise neutral. rho is of the REWARD with heavy atoms (reward up = better)."""
    if np.isnan(t[0]) or abs(t[0]) < 0.15 or t[1] * t[2] <= 0:
        return "neutral"
    return f"toward {sign_name[0] if t[0] < 0 else sign_name[1]}"


def table(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pnorm(x, lo, hi, minimize):
    z = np.clip((np.asarray(x, float) - lo) / max(hi - lo, 1e-6), 0, 1)
    return 1 - z if minimize else z


def analyse():
    import generation_finetuning as gf
    df = pd.read_csv(f"{OUT}/molecules.csv")
    n_raw = len(df)
    df = df.drop_duplicates("canon", keep="first").reset_index(drop=True)          # source order: baseline, fresh, probe archives
    df["group"] = np.where(df.src.isin(PROBES), "archive", "prior")
    H = df.heavy.values.astype(float)
    th = {s: mo_mpo.get_constraint_thresholds(s) for s in (1, 2, 3)}
    w_new, w_old = mo_mpo.get_curriculum_weights(2), OLD_STAGE2
    groups = {"all": np.ones(len(df), bool), "prior": (df.group == "prior").values, "archive": (df.group == "archive").values}
    P = groups["prior"]
    docked = df.docking.notna().values

    def sc(i):
        return {k: (None if pd.isna(df.at[i, k]) else float(df.at[i, k])) for k in OBJ7}
    scs = [sc(i) for i in range(len(df))]
    passes = {s: np.array([not (mo_mpo.is_unscored(x) or mo_mpo.violates_hard_constraints(x, th[s])) for x in scs]) for s in (1, 2, 3)}
    gate_ok = passes[2]
    cc = df[OBJ6].notna().all(axis=1).values                                                            # complete cases for a combined reward

    # two normalisations: the RL Normalizer at its initial bounds, and at the bounds it converges to (P5/P95 of the scored = gate-passing, unselected molecules)
    nm = gf.Normalizer()
    bnd = {n: np.nanpercentile(df[n][P & gate_ok], [5, 95]) for n in OBJ7}

    def rewards(kind):
        R = pd.DataFrame({n: [(nm(n, v) if kind == "initial" else float(pnorm(v, *bnd[n], nm.obj[n].minimize))) if pd.notna(v) else np.nan for v in df[n]] for n in OBJ7})
        R["comb_new"] = sum(w_new[n] * R[n] for n in OBJ6)
        R["comb_old"] = sum(w_old[n] * R[n] for n in OBJ6) + w_old["steric"] * R["steric"]
        R.loc[~cc, ["comb_new", "comb_old"]] = np.nan
        R["rl"] = np.where(gate_ok & docked & cc, R.comb_new.fillna(0), 0.0)                            # RL reward of a valid molecule: gate fail / dock fail = 0
        return R
    RI, RA = rewards("initial"), rewards("adapted")
    df["rl_reward"] = RI.rl
    out = {}
    md = [f"# Size-bias audit\n\nMolecules: {n_raw} collected, {len(df)} after de-duplication by canonical SMILES. prior = baseline 93 + 93 valid of 256 fresh prior samples "
          f"(unselected, n={P.sum()}); archive = Pareto archives of the 3 probes (selected survivors, n={groups['archive'].sum()}). Sources: {df.src.value_counts().to_dict()}. "
          f"Heavy atoms: mean {H.mean():.1f} (prior {H[P].mean():.1f}, archive {H[groups['archive']].mean():.1f}). "
          f"Bounds: 'initial' = the RL Normalizer's starting ranges, 'adapted' = P5/P95 of the {int((P & gate_ok).sum())} gate-passing prior molecules (where the Normalizer drifts to, momentum 0.99). "
          "Spearman is rank-based, so the per-objective correlations are the same under both; only combined rewards depend on the bounds.\n"]

    # ---- 2. Spearman of each objective reward with heavy atoms
    rows, out["spearman"] = [], {}
    names = {"comb_new": "COMBINED, stage 2 NEW weights", "comb_old": "COMBINED, stage 2 weights of the probes", "steric": "steric (weight 0 now, gate only)"}
    for c in OBJ6 + ["steric", "comb_new", "comb_old"]:
        t = {g: rho(H[m], RI[c].values[m]) for g, m in groups.items()}
        m = cc & P
        slopes = [slope10(H[m], R[c].values[m]) * (w_new[c] if c in OBJ6 else 1) if (c in OBJ6 or c.startswith("comb")) else np.nan for R in (RI, RA)]
        out["spearman"][c] = {g: list(map(float, v)) for g, v in t.items()} | {"weighted_slope_per10_prior": [None if np.isnan(s) else float(s) for s in slopes]}
        rows.append([names.get(c, c), fmt_rho(t["all"]), fmt_rho(t["prior"]), fmt_rho(t["archive"])] + ["" if np.isnan(s) else f"{s:+.3f}" for s in slopes] + [verdict(t["prior"]), verdict(t["all"])])
    t = {g: rho(H[m], RI.rl.values[m]) for g, m in groups.items()}
    s = [slope10(H[P], R_.rl.values[P]) for R_ in (RI, RA)]
    out["spearman"]["rl_reward"] = {g: list(map(float, v)) for g, v in t.items()} | {"slope_per10_prior": s}
    rows.append(["RL reward of a valid molecule (gate or dock fail = 0), NEW st. 2", fmt_rho(t["all"]), fmt_rho(t["prior"]), fmt_rho(t["archive"])] + [f"{x:+.3f}" for x in s] + [verdict(t["prior"]), verdict(t["all"])])
    md += ["## 2. Spearman of the normalised reward with heavy atoms (reward up = better, so a NEGATIVE rho = the term favours SMALLER molecules)\n",
           table(rows, ["objective", "all [95% CI]", "prior only (unselected)", "archives only (selected)", "weight x slope per +10 atoms, prior, initial bounds",
                        "same, adapted bounds", "verdict (prior)", "verdict (all)"]),
           "\nVerdict rule: |rho| >= 0.15 and a bootstrap CI that excludes 0. 'weight x slope' = the term's contribution to the change of the combined NEW stage-2 reward "
           "per +10 heavy atoms (OLS on the complete-case prior molecules; the six terms add up to the COMBINED row; the RL-reward row is the slope of the reward with gate fails = 0).\n"]
    bins, bl_ = [0, 14, 19, 24, 29, 999], ["<=14", "15-19", "20-24", "25-29", ">=30"]
    hb = pd.cut(df.heavy, bins, labels=bl_).astype(str).values
    rows = [[lb, int((P & (hb == lb)).sum())] + [f"{np.nanmean(RI[c][P & (hb == lb)]):.2f}" for c in OBJ6 + ["comb_new"]] + [f"{RI.rl[P & (hb == lb)].mean():.2f}"] for lb in bl_]
    md += ["Mean normalised reward by heavy-atom bin (prior only, initial bounds):\n", table(rows, ["heavy atoms", "n"] + OBJ6 + ["combined (scored)", "RL reward (gate fail = 0)"]), "\n"]

    # ---- 3. gate pass rate by heavy-atom tercile
    def gate_block(m, label):
        d, hh = df[m].reset_index(drop=True), H[m]
        terc = np.digitize(hh, np.quantile(hh, [1 / 3, 2 / 3]), right=True)
        lab = [f"T{i + 1} ({int(hh[terc == i].min())}-{int(hh[terc == i].max())} atoms, n={int((terc == i).sum())})" for i in range(3)]
        t2 = th[2]
        g = {"3D build ok": (d.torsion.notna() & d.energy.notna() & d.steric.notna()).values, "energy <= %g" % t2["energy_max"]: (d.energy <= t2["energy_max"]).values,
             "torsion <= %g" % t2["torsion_max"]: (d.torsion <= t2["torsion_max"]).values, "SA <= %.3f" % t2["sa_max"]: (d.sa <= t2["sa_max"]).values,
             "steric <= %g" % t2["steric_max"]: (d.steric <= t2["steric_max"]).values, "ALL cheap gates, stage 2": passes[2][m], "all cheap gates, stage 1": passes[1][m],
             "all cheap gates, stage 3": passes[3][m], "stage 2 gates and docking ok": passes[2][m] & docked[m]}
        rows = [[k] + [f"{100 * np.mean(v[terc == i]):.0f}%" for i in range(3)] + [f"{100 * np.mean(v):.0f}%"] for k, v in g.items()]
        return f"**{label}**\n\n" + table(rows, ["gate (pass rate)"] + lab + ["all"]) + "\n", {k: [float(np.mean(v[terc == i])) for i in range(3)] for k, v in g.items()}
    md += ["## 3. Gate pass rate by heavy-atom tercile (stage 2 gates unless stated; per-gate rates are over all valid molecules, a missing 3D build counts as a fail; "
           "the archives passed their gates by construction, so the unselected prior is the relevant block)\n"]
    out["gates"] = {}
    for label, key in (("unselected prior samples", "prior"), ("all molecules including archives", "all")):
        s, out["gates"][key] = gate_block(groups[key], label)
        md.append(s)

    # ---- 4. validity vs token length (prior samples: baseline 256 on targets 0 and 3 + 256 fresh on target 0)
    sm = pd.read_csv(f"{OUT}/samples.csv")
    canon_rl = dict(zip(df.canon, df.rl_reward))
    sm["rl"] = [(-1.0 if not v else canon_rl.get(Chem.MolToSmiles(Chem.MolFromSmiles(s)), 0.0)) for s, v in zip(sm.smiles.fillna(""), sm.valid)]
    sm["bin"] = pd.qcut(sm.n_tokens, 8, duplicates="drop")
    rows = []
    for b, g in sm.groupby("bin", observed=True):
        term = g[g.category != "no_termination"]
        rows.append([f"{int(b.left) + 1}-{int(b.right)}", len(g), f"{100 * g.valid.mean():.0f}%", f"{100 * (g.category == 'no_termination').mean():.0f}%",
                     f"{100 * term.valid.mean():.0f}%" if len(term) else "", f"{g.rl.mean():+.2f}"])
    term_all = sm[sm.category != "no_termination"]
    r_all, r_term, r_rl = rho(sm.n_tokens, sm.valid.astype(float)), rho(term_all.n_tokens, term_all.valid.astype(float)), rho(sm.n_tokens, sm.rl)
    out["validity_vs_length"] = dict(rho_all=list(map(float, r_all)), rho_terminated=list(map(float, r_term)), rho_rl_reward=list(map(float, r_rl)), bins=[[str(c) for c in r] for r in rows])
    md += [f"## 4. Validity vs token length ({len(sm)} prior samples, 8 equal-count bins)\n",
           table(rows, ["tokens", "n", "valid", "never terminated (hit the 200-token budget)", "valid among terminated", "mean RL reward (invalid -1, gate fail 0, stage 2 NEW)"]),
           f"\nSpearman of token length with validity: all samples {fmt_rho(r_all)}; terminated samples only {fmt_rho(r_term)}; "
           f"token length with the RL reward (invalid = -1): {fmt_rho(r_rl)}.\n"]

    # ---- 5. docking and ligand efficiency vs heavy atoms
    df["lig_eff"] = df.docking / df.heavy
    rows, out["docking_le"] = [], {}
    for g, m in groups.items():
        for col, lab in (("docking", "docking (kcal/mol)"), ("lig_eff", "ligand efficiency (kcal/mol/atom)")):
            ok = m & df[col].notna().values
            t, lr = rho(H[ok], df[col].values[ok]), linregress(H[ok], df[col].values[ok])
            out["docking_le"][f"{g}_{col}"] = dict(n=int(ok.sum()), slope_per_atom=float(lr.slope), rho=list(map(float, t)), pearson=float(lr.rvalue))
            rows.append([g, lab, int(ok.sum()), f"{lr.slope:+.4f}", fmt_rho(t), f"{lr.rvalue:+.2f}"])
    md += ["## 5. Docking and ligand efficiency vs heavy atoms (docking is negative = better; ligand efficiency = docking / heavy atoms, also negative = better)\n",
           table(rows, ["set", "quantity", "n", "OLS slope per heavy atom", "Spearman [95% CI]", "Pearson"]), "\n"]
    rows = []
    for lb in bl_:
        m = hb == lb
        rows.append([lb, int(m.sum()), f"{df.docking[m].mean():.2f}", f"{df.lig_eff[m].mean():.3f}", f"{100 * gate_ok[m & P].mean():.0f}%", f"{df.rl_reward[m & P].mean():.2f}"])
    md += ["Mean by heavy-atom bin (docking and ligand efficiency: all molecules; the last two columns: prior only):\n",
           table(rows, ["heavy atoms", "n", "docking", "ligand efficiency", "stage 2 gate pass", "RL reward"]), "\n"]

    # ---- 6. what the candidate fixes would change (hypothetical rewards on the same molecules; nothing is implemented)
    pool = df[P & gate_ok]
    nrm_pa = {n: np.nanpercentile(pool[n] / pool.heavy, [5, 95]) for n in ("energy", "torsion")}
    nrm_le = np.nanpercentile(pool.docking / pool.heavy, [5, 95])
    base = {n: RA[n].values for n in OBJ6}
    A = {**base, **{n: pnorm((df[n] / df.heavy).values, *nrm_pa[n], True) for n in nrm_pa}}
    C = {**base, "docking": pnorm(df.lig_eff.values, *nrm_le, True)}
    variants = {"current terms": base, "(A) energy and torsion per heavy atom": A, "(C) docking as ligand efficiency": C, "(A)+(C)": {**A, "docking": C["docking"]}}
    rows, out["fix_test"] = [], {}
    for name, v in variants.items():
        comb = sum(w_new[n] * v[n] for n in OBJ6)
        r_ = [rho(H[m & cc], comb[m & cc]) for m in (P, groups["all"])]
        sl = [slope10(H[m & cc], comb[m & cc]) for m in (P, groups["all"])]
        out["fix_test"][name] = dict(rho_prior=list(map(float, r_[0])), rho_all=list(map(float, r_[1])), slope_prior=float(sl[0]), slope_all=float(sl[1]))
        rows.append([name, fmt_rho(r_[0]), f"{sl[0]:+.3f}", fmt_rho(r_[1]), f"{sl[1]:+.3f}"])
    md += ["## 6. Candidate fixes tested offline (combined NEW stage-2 reward of the scored molecules vs heavy atoms; adapted bounds, per-atom/LE bounds = P5/P95 of the same gate-passers)\n",
           table(rows, ["reward variant", "prior: Spearman [95% CI]", "prior: slope per +10 atoms", "all: Spearman", "all: slope"]),
           "\nPositive = the combined reward favours LARGER molecules.\n"]
    rows = []
    for n, x in (("UFF energy", df.energy), ("torsion strain", df.torsion), ("energy per heavy atom", df.energy / df.heavy), ("torsion per heavy atom", df.torsion / df.heavy),
                 ("steric", df.steric), ("SA", df.sa), ("QED", df.qed)):
        t = rho(H[P], x.values[P]), rho(H, x.values)
        out["fix_test"]["raw_" + n] = [list(map(float, u)) for u in t]
        rows.append([n, fmt_rho(t[0]), fmt_rho(t[1])])
    ks = np.arange(0, 2.01, 0.1)
    neutral_k = float(ks[np.argmin([abs(spearmanr(H[P], (df.energy / df.heavy ** k).values[P], nan_policy='omit')[0]) for k in ks])])
    md += ["Raw value vs heavy atoms (+ = grows with size):\n", table(rows, ["quantity", "prior: Spearman [95% CI]", "all"]),
           f"\nUFF energy / heavy_atoms^k is size-neutral in the prior at k = {neutral_k:.1f} (k = 1 is 'per heavy atom').\n"]
    t2, rows = th[2], []
    terc = np.digitize(H[P], np.quantile(H[P], [1 / 3, 2 / 3]), right=True)
    for n, cur in (("energy", t2["energy_max"]), ("torsion", t2["torsion_max"])):
        v = (df[n] / df.heavy).values[P]
        okv = df[n].notna().values[P]
        rate = np.mean(df[n].values[P][okv] <= cur)
        thr = np.quantile(v[okv], rate)
        cur_pass, pa_pass = df[n].values[P] <= cur, v <= thr
        rows.append([f"{n} gate", f"{cur:g} (pass {100 * rate:.0f}%)"] + [f"{100 * np.mean(cur_pass[terc == i]):.0f}% -> {100 * np.mean(pa_pass[terc == i]):.0f}%" for i in range(3)])
    md += ["(A) at the gate: stage 2 pass rate by size tercile (prior), current threshold -> a per-atom threshold with the same overall pass rate:\n",
           table(rows, ["gate", "current threshold", "T1 (small)", "T2", "T3 (large)"]), "\n"]
    rows = []
    for k in (12, 15, 18, 20):
        small = H < k
        rows.append([f"< {k}", f"{100 * small[P].mean():.0f}%", f"{100 * small[P & gate_ok].mean():.0f}%", f"{100 * small[groups['archive']].mean():.0f}%",
                     f"{df.rl_reward[P & small].mean():.2f}", f"{df.rl_reward[P & ~small].mean():.2f}"])
    md += ["(B) share of molecules below a heavy-atom floor and the RL reward (stage 2 NEW, prior) they receive:\n",
           table(rows, ["heavy atoms", "prior samples", "prior gate-passers", "probe archives", "RL reward below", "RL reward at or above"]), "\n"]
    open(f"{OUT}/audit.md", "w").write("\n".join(md))
    json.dump(out, open(f"{OUT}/audit.json", "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("collect", "all"):
        collect()
    if what in ("analyse", "all"):
        analyse()
