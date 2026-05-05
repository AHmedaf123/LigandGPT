# LigandGPT: Curriculum-Guided Reinforcement Learning for Molecular Design


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of **LigandGPT**, a curriculum-guided reinforcement learning framework for structure-based molecular design.

## Abstract

Use of Generative AI for the design of new drugs with computers is inherently difficult due to its capacity to generate molecules that are either too unstable or too difficult to construct within a real lab setting. Current methods typically center on the extent to which molecules adhere to a given target protein, fixing physical problems such as steric clashes or torsional strain only after the molecule is constructed. LigandGPT seeks to address this by integrating curriculum-guided Multi-Objective Optimization with policy-level enforcement of physical constraints on molecular formation, applying these progressively during generation to ensure that generated candidates are structurally consistent and experimentally viable. LigandGPT thus comprises a unified pipeline of three components: a Graph Neural Network (GNN) surrogate to predict bioactivity, an autoregressive transformer to generate ligand sequences, and curriculum-guided reinforcement learning with policy-level physical constraints encompassing steric clash, torsional strain and internal energy. Policy Optimisation adopts a fixed-prior KL-anchored approach to prevent policy drift away from valid chemical syntax along with Tanimoto similarity-based novelty regularization. Archive selection employs Augmented Tchebycheff Scalarization to improve coverage in non-convex regions thereby avoiding weakly Pareto-optimal solutions. Results indicate that LigandGPT, in encompassing both physical and practical laws as it builds molecules, achieves best in class performance on standard benchmarks, giving rise to molecules that are 61% more stable and 88% more realistic than comparable leading models, and hence drug candidates that are proportionately more likely to succeed in real-world experiments.

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
