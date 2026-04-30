import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm
from torch.nn import LayerNorm
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.conv = GINConv(self.mlp, train_eps=True)
        self.norm = BatchNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x, edge_index, edge_attr=None):
        identity = self.residual(x)
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x + identity)
        return x
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, dropout=0.1, edge_dim=None):
        super().__init__()
        self.conv = GATConv(
            in_dim, 
            out_dim // heads, 
            heads=heads, 
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
            concat=True
        )
        self.norm = LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x, edge_index, edge_attr=None):
        identity = self.residual(x)
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = F.elu(x + identity)
        return x

class ProteinPocketEncoder(nn.Module):
    """Encodes protein pocket atom features into a fixed-size representation.
    
    Implements f_prot: {atom_features} -> R^{prot_hidden_dim} via a DeepSets-style 
    architecture: per-atom MLP followed by permutation-invariant pooling.
    
    This enables the bioactivity predictor to compute y_bio = f(P, L) as specified
    in the paper's equation [B3], rather than y_bio = f(L).
    """
    def __init__(self, prot_input_dim=40, prot_hidden_dim=128, dropout=0.15):
        super().__init__()
        # Per-atom feature encoder (phi in DeepSets)
        self.atom_encoder = nn.Sequential(
            nn.Linear(prot_input_dim, prot_hidden_dim),
            nn.BatchNorm1d(prot_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prot_hidden_dim, prot_hidden_dim),
            nn.BatchNorm1d(prot_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Post-aggregation transform (rho in DeepSets)
        self.pool_transform = nn.Sequential(
            nn.Linear(prot_hidden_dim * 2, prot_hidden_dim),
            nn.LayerNorm(prot_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.prot_hidden_dim = prot_hidden_dim
    
    def forward(self, prot_x, prot_batch):
        """
        Args:
            prot_x: [total_prot_atoms, prot_input_dim] - protein pocket atom features
            prot_batch: [total_prot_atoms] - batch assignment for each protein atom
        
        Returns:
            prot_repr: [num_graphs, prot_hidden_dim] - per-graph protein representation
        """
        # Per-atom encoding
        h = self.atom_encoder(prot_x)
        
        # Permutation-invariant aggregation (mean + max pooling)
        h_mean = global_mean_pool(h, prot_batch)  # [num_graphs, prot_hidden_dim]
        h_max = global_max_pool(h, prot_batch)     # [num_graphs, prot_hidden_dim]
        
        # Combine pooled representations
        h_pool = torch.cat([h_mean, h_max], dim=1)  # [num_graphs, prot_hidden_dim * 2]
        prot_repr = self.pool_transform(h_pool)       # [num_graphs, prot_hidden_dim]
        
        return prot_repr


class AdvancedGNN(nn.Module):
    def __init__(
        self,
        node_input_dim=75,
        edge_input_dim=12,
        hidden_dim=256,
        num_gin_layers=5,
        num_gat_layers=2,
        gat_heads=8,
        dropout=0.15,
        num_tasks=1,
        prot_input_dim=40,
        prot_hidden_dim=128
    ):
        super().__init__()
        # --- Ligand encoder (unchanged) ---
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ) if edge_input_dim > 0 else None
        self.gin_layers = nn.ModuleList()
        for i in range(num_gin_layers):
            self.gin_layers.append(
                GINLayer(hidden_dim, hidden_dim, dropout=dropout)
            )
        self.gat_layers = nn.ModuleList()
        gat_dim = hidden_dim
        for i in range(num_gat_layers):
            self.gat_layers.append(
                GATLayer(
                    gat_dim, 
                    hidden_dim, 
                    heads=gat_heads, 
                    dropout=dropout,
                    edge_dim=hidden_dim if self.edge_encoder else None
                )
            )
        self.virtual_node_mlp = nn.ModuleList()
        for i in range(num_gin_layers + num_gat_layers):
            self.virtual_node_mlp.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
            )
        
        # --- Protein pocket encoder [B3: y_bio = f(P, L)] ---
        self.protein_encoder = ProteinPocketEncoder(
            prot_input_dim=prot_input_dim,
            prot_hidden_dim=prot_hidden_dim,
            dropout=dropout
        )
        self.prot_hidden_dim = prot_hidden_dim
        
        # --- Predictor head: now takes ligand_pool + protein_repr ---
        self.pool_dim = hidden_dim * 3  # ligand: mean + max + add pooling
        predictor_input_dim = self.pool_dim + prot_hidden_dim  # ligand + protein
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),            
            nn.Linear(hidden_dim // 2, num_tasks)
        )        
        self.hidden_dim = hidden_dim
        self.dropout = dropout 

    def _encode_ligand(self, batch):
        """Encode ligand molecular graph → pooled representation."""
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.node_encoder(x)
        if self.edge_encoder and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        num_graphs = batch_idx.max().item() + 1
        virtual_node = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        layer_outputs = []
        for i, gin_layer in enumerate(self.gin_layers):
            x = x + virtual_node[batch_idx]
            x = gin_layer(x, edge_index, edge_attr)
            layer_outputs.append(x)
            virtual_node_temp = global_add_pool(x, batch_idx)
            virtual_node = virtual_node + F.dropout(
                self.virtual_node_mlp[i](virtual_node_temp),
                p=self.dropout,
                training=self.training
            )
        for i, gat_layer in enumerate(self.gat_layers):
            x = x + virtual_node[batch_idx]
            x = gat_layer(x, edge_index, edge_attr)
            layer_outputs.append(x)
            virtual_node_temp = global_add_pool(x, batch_idx)
            virtual_node = virtual_node + F.dropout(
                self.virtual_node_mlp[len(self.gin_layers) + i](virtual_node_temp),
                p=self.dropout,
                training=self.training
            )
        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        x_add = global_add_pool(x, batch_idx)
        x_pool = torch.cat([x_mean, x_max, x_add], dim=1)
        return x_pool, num_graphs

    def _encode_protein(self, batch, num_graphs):
        """Encode protein pocket features → pooled representation.
        
        Falls back to zero vector if protein features are not provided,
        maintaining backward compatibility for ligand-only inference.
        """
        if hasattr(batch, 'prot_x') and batch.prot_x is not None and batch.prot_x.numel() > 0:
            prot_batch = batch.prot_batch if hasattr(batch, 'prot_batch') else torch.zeros(
                batch.prot_x.size(0), dtype=torch.long, device=batch.prot_x.device
            )
            prot_repr = self.protein_encoder(batch.prot_x, prot_batch)
        else:
            # Fallback: zero protein representation for backward compatibility
            device = batch.x.device
            prot_repr = torch.zeros(num_graphs, self.prot_hidden_dim, device=device)
        return prot_repr

    def forward(self, batch):
        # Encode ligand graph
        x_pool, num_graphs = self._encode_ligand(batch)
        
        # Encode protein pocket [B3: y_bio = f(P, L)]
        prot_repr = self._encode_protein(batch, num_graphs)
        
        # Fuse ligand + protein representations
        fused = torch.cat([x_pool, prot_repr], dim=1)  # [num_graphs, pool_dim + prot_hidden_dim]
        
        out = self.predictor(fused)
        return out

    def get_embeddings(self, batch):
        x_pool, num_graphs = self._encode_ligand(batch)
        prot_repr = self._encode_protein(batch, num_graphs)
        embeddings = torch.cat([x_pool, prot_repr], dim=1)
        return embeddings

def create_model(config=None):
    """Factory function to create model with config"""
    if config is None:
        config = {
            'node_input_dim': 75,
            'edge_input_dim': 12,
            'hidden_dim': 256,
            'num_gin_layers': 5,
            'num_gat_layers': 2,
            'gat_heads': 8,
            'dropout': 0.15,
            'num_tasks': 1,
            'prot_input_dim': 40,
            'prot_hidden_dim': 128
        }
    return AdvancedGNN(**config)