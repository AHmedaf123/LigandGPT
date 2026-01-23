# LigandGPT: Curriculum-Guided Reinforcement Learning for Molecular Design


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of **LigandGPT**, a curriculum-guided reinforcement learning framework for structure-based molecular design.

## Abstract

Most deep learning models for structure-based drug design focus only on binding affinity. This 
creates problems. Generated molecules are often unstable, lack drug-like properties, or are too hard to synthesize. This limits their real-world use. We present LigandGPT, a framework that solves this issue. It optimizes seven properties at once: binding affinity, bioactivity, drug-likeness,synthetic accessibility, and three stability measures (steric clash, torsional strain, and internal energy). The system uses a 12-layer transformer to generate molecules. A GNN-based model and QVina2.0 evaluate binding. RDKit provides physicochemical scores. Multi-objective learning faces reward interference and mode collapse. We solve this with a three-stage curriculum. It activates objectives gradually and uses Chebyshev scalarization for optimal balancing. This reduces reward variance and improves stability. Unlike previous methods that fix problems after generation, LigandGPT enforces physical constraints during generation. Combined with experience replay and novelty exploration, it converges faster and produces molecules with lower strain while maintaining strong binding affinity. Our results show that enforcing constraints during generation works better than post-hoc corrections for stable molecular design. 

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
