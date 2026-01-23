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
        num_tasks=1
    ):
        super().__init__()
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
        self.pool_dim = hidden_dim * 3
        self.predictor = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim * 2),
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
    def forward(self, batch):
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
        out = self.predictor(x_pool)
        return out
    def get_embeddings(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.node_encoder(x)
        if self.edge_encoder and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)        
        num_graphs = batch_idx.max().item() + 1
        virtual_node = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        for i, gin_layer in enumerate(self.gin_layers):
            x = x + virtual_node[batch_idx]
            x = gin_layer(x, edge_index, edge_attr)
            virtual_node_temp = global_add_pool(x, batch_idx)
            virtual_node = virtual_node + self.virtual_node_mlp[i](virtual_node_temp)       
        for i, gat_layer in enumerate(self.gat_layers):
            x = x + virtual_node[batch_idx]
            x = gat_layer(x, edge_index, edge_attr)
            virtual_node_temp = global_add_pool(x, batch_idx)
            virtual_node = virtual_node + self.virtual_node_mlp[len(self.gin_layers) + i](virtual_node_temp)       
        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        x_add = global_add_pool(x, batch_idx)        
        embeddings = torch.cat([x_mean, x_max, x_add], dim=1)
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
            'num_tasks': 1
        }
    return AdvancedGNN(**config)