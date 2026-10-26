import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GINEConv


class NodeNorm(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.norm, (nn.InstanceNorm1d, nn.GroupNorm)) and x.dim() == 2:
            x = x.unsqueeze(-1)
            x = self.norm(x)
            return x.squeeze(-1)
        return self.norm(x)


def make_node_norm(norm_type: str, dim: int, eps: float) -> nn.Module:
    norm_type = str(norm_type).strip().lower()
    if norm_type in ("batch", "batchnorm", "bn"):
        return nn.BatchNorm1d(dim, eps=eps)
    if norm_type in ("layer", "layernorm", "ln"):
        return nn.LayerNorm(dim, eps=eps)
    if norm_type in ("instance", "instancenorm", "in"):
        return NodeNorm(nn.GroupNorm(1, dim, eps=eps, affine=True))
    raise ValueError(f"Unsupported norm_type='{norm_type}'. Use batch|layer|instance.")


class ScenarioGCN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        edge_attr_dim: int = 2,
        norm_type: str = "batch",
    ):
        super().__init__()

        # 1. Initial Embedding
        self.emb_mlp = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            make_node_norm(norm_type, hidden_dim, eps=1e-4),
            nn.ReLU()
        )

        self.edge_proj = nn.Linear(edge_attr_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()  # <--- Standard Norm list

        dims = [hidden_dim] * num_layers + [out_dim]

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            # MLP for GINEConv
            mlp = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.ReLU(),
                nn.Linear(d_out, d_out)
            )
            # train_eps=True learns the (1+eps) factor automatically
            self.convs.append(GINEConv(mlp, train_eps=True))

            self.norms.append(make_node_norm(norm_type, d_out, eps=1e-4))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr, scen_batch):
        # 1. Initial Projection
        x = self.emb_mlp(x)
        x = self.dropout(x)
        edge_emb = self.edge_proj(edge_attr)

        # 2. Convolutions
        for conv, norm in zip(self.convs, self.norms):
            # GINEConv
            x = conv(x, edge_index, edge_attr=edge_emb)

            # Apply Standard Batch Norm
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        return global_mean_pool(x, scen_batch)



