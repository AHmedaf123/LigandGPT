"""Confidence intervals, kekulize-failure structure and type-violation count for results/generation_diagnosis/samples.csv.
Run: .venv/bin/python analyze_generation_diagnosis.py > results/generation_diagnosis/analysis.txt"""
import csv, collections, math, sys
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
rows=list(csv.DictReader(open("results/generation_diagnosis/samples.csv")))
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
print("setting                 strict   95% Wilson CI     | per-target")
for s in dict.fromkeys(r["setting"] for r in rows):
    R=[r for r in rows if r["setting"]==s]; k=sum(int(r["strict_valid"]) for r in R); lo,hi=wilson(k,len(R))
    pt=[f"T{t}:{sum(int(r['strict_valid']) for r in R if r['target']==t)}/{sum(1 for r in R if r['target']==t)}" for t in ("0","3")]
    print(f"{s:22s} {k:3d}/{len(R)}={k/len(R):.3f}  [{lo:.3f}, {hi:.3f}] | {' '.join(pt)}")
print("\nkekulize failures: structure of the aromatic atoms")
for s in ("plain_k10","current_k10","struct_syntax_k10","struct_valence_k10"):
    R=[r for r in rows if r["setting"]==s and r["category"].startswith("rdkit:kekulize")]; c=collections.Counter(); ex={}
    for r in R:
        m=Chem.MolFromSmiles(r["smiles"],sanitize=False); m.UpdatePropertyCache(strict=False); Chem.FastFindRings(m)
        ri=m.GetRingInfo(); arom=[a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()]
        inring=[i for i in arom if ri.NumAtomRings(i)>0]
        if len(inring)<len(arom): k="aromatic atom outside any ring"
        else:
            rings=[x for x in ri.AtomRings() if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in x)]
            sizes=sorted(len(x) for x in rings)
            if any(z not in (5,6) for z in sizes): k="aromatic ring size not 5/6 (%s)"%",".join(str(z) for z in sorted(set(sizes)-{5,6}))
            elif any(z==5 for z in sizes):
                def donor(x): return any(m.GetAtomWithIdx(i).GetSymbol() in "OSos" or m.GetAtomWithIdx(i).GetNumExplicitHs()>0 for i in x)
                k="5-ring aromatic without donor (o/s/[nH])" if any(len(x)==5 and not donor(x) for x in rings) else "5/6-ring electron count wrong"
            else: k="6-ring electron count wrong (e.g. odd number of pi atoms)"
        c[k]+=1; ex.setdefault(k,r["smiles"])
    print(f"-- {s}: {len(R)} kekulize failures"); [print(f"   {v:3d}  {k}   e.g. {ex[k][:70]}") for k,v in c.most_common()]
print("\nvalence-mode settings: failures by category (pooled 128)")
for s in ("struct_valence_k10","struct_valence_full"):
    c=collections.Counter(r["category"] for r in rows if r["setting"]==s); print(" ",s,dict(c))
tv=[r for r in rows if r["category"].startswith("type_violation")]; print("\ntype_violation samples across ALL settings:",len(tv),"of",len(rows))
