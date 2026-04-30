# LigandGPT: Curriculum-Guided Reinforcement Learning for Molecular Design


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of **LigandGPT**, a curriculum-guided reinforcement learning framework for structure-based molecular design.

## Abstract

Designing new drugs with computers is difficult because AI often creates ``impossible'' molecules that are unstable or too hard to build in a real lab. Current methods usually focus only on how well a molecule sticks to a target protein and try to fix physical problems after the molecule is already finished. \textbf{LigandGPT} solves this by using curriculum-based learning that follows a three-stage training pipeline. First, the AI learns the basics of how to make a molecule stick to a protein. Next, it learns how to make the molecule look more like a real drug. Finally, it learns to balance seven different objectives simultaneously docking score, bioactivity, drug-likeness, synthetic accessibility, steric clash, torsional strain, and molecular energy to generate optimized ligand candidates. Instead of fixing mistakes at the end, LigandGPT follows physical laws while it builds the molecule to ensure the structure is not too twisted or unstable. Results show that LigandGPT outperforms top AI benchmarks. Specifically, it creates molecules that are 61\% more stable and 88\% more realistic than other leading models. This approach helps scientists find drug candidates that are much more likely to succeed in real-world experiments.

## Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| Uni-Mol | LigandGPT 3D pre-training | [GitHub](https://github.com/deepmodeling/Uni-Mol) |
| CrossDocked2020 | Protein–ligand pair pre-training | [GitHub](https://github.com/gnina/models) |
| ChEMBL | AdvanceGNN pre-training | [ChEMBL](https://www.ebi.ac.uk/chembl/) |
| PDBbind v2020 | AdvanceGNN fine-tuning (binding affinity) | [PDBbind](https://www.pdbbind-plus.org.cn/) |
| TargetDiff split | RL generation benchmarks | [GitHub](https://github.com/guanjq/targetdiff) |

## Usage

### LigandGPT Pre-training
```bash
python pretraining.py --dataset_path ./data/ --ckpt_save_path ./checkpoints/
```

### AdvanceGNN Pre-training (ChEMBL)
```bash
python chEMBL-pretraining.py
```

### AdvanceGNN Fine-tuning (PDBbind)
```bash
python PDBbind-finetuning.py --pretrained_path ./outputs/chembl_pretrain/best_model.pt
```

### RL Fine-tuning (Molecule Generation)
```bash
python generation_finetuning.py --ckpt_load_path ./checkpoints/final.pt --n_steps 700
```

## Model Architecture

| Component | Details |
|-----------|---------|
| LigandGPT | 12 layers, 12 heads, 768-dim (~85M params) |
| AdvanceGNN (bioactivity GNN) | 5 GIN + 2 GAT layers (~5M params) |
