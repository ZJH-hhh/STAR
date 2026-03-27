import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing


class RBF(nn.Module):
    """Radial basis feature expansion."""

    def __init__(self, centers, gamma):
        super().__init__()
        centers = torch.as_tensor(centers, dtype=torch.float32).view(1, -1)
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class BondFloatRBF(nn.Module):
    """Encode continuous edge weights (e.g., scaffold/group similarities)."""

    def __init__(self, embed_dim, centers=None, gamma=10.0):
        super().__init__()
        if centers is None:
            centers = torch.arange(0, 2, 0.1)
        self.rbf = RBF(centers, gamma)
        self.linear = nn.Linear(len(centers), embed_dim)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        return self.linear(self.rbf(weights))


class HMRGNNConv(MessagePassing):
    """Single HGNN block that mixes neighbouring embeddings with edge attrs."""

    def __init__(self, input_dim, output_dim, norm=False, batch_norm=False, dropout=0.0):
        super().__init__(aggr="mean")
        self.concat_linear = nn.Linear(input_dim * 2, output_dim)
        self.root_linear = nn.Linear(input_dim, output_dim)
        self.norm = norm
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_embeddings):
        msg = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        msg = msg + self.root_linear(x)
        if self.norm:
            msg = F.normalize(msg, p=2, dim=-1)
        if self.batch_norm:
            msg = self.batch_norm(msg)
        return self.dropout(self.act(msg))

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class HMRGNN(nn.Module):
    """Heterogeneous relation GNN copied from KRGTS (Stage B backend)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_edge_type,
        num_edge_attr_int,
        num_edge_attr_float,
        num_relation_attr,
        batch_norm=False,
        dropout=0.0,
        norm=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.edge_type_emb = nn.Embedding(num_edge_type + 1, hidden_dim)
        self.int_edge_attr = nn.ModuleList(
            [nn.Embedding(num_embeddings=num_relation_attr, embedding_dim=hidden_dim) for _ in range(num_edge_attr_int)]
        )
        self.float_edge_attr = nn.ModuleList(
            [BondFloatRBF(hidden_dim) for _ in range(num_edge_attr_float)]
        )
        self.edge_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.norm = norm
        self.batch_norm = batch_norm

        blocks = []
        for _ in range(num_edge_type - 1):
            for _ in range(num_layers):
                blocks.append(
                    HMRGNNConv(
                        input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        norm=norm,
                        batch_norm=batch_norm,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.out_linear = nn.Linear(hidden_dim * 2, output_dim)
        self.out_bn = nn.BatchNorm1d(output_dim) if batch_norm else None

    def forward(self, x, edge_index, edge_type_tokens, edge_type_list, edge_weight):
        prop_mask = ~(edge_type_list - edge_type_tokens[0]).bool()
        sca_mask = ~(edge_type_list - edge_type_tokens[1]).bool()
        gro_mask = ~(edge_type_list - edge_type_tokens[2]).bool()

        type_embed = self.edge_type_emb(edge_type_list)
        attr_embed = type_embed.clone()
        if prop_mask.any():
            attr_embed[prop_mask] = self.int_edge_attr[0](edge_weight[prop_mask, 0].long())
        if sca_mask.any():
            attr_embed[sca_mask] = self.float_edge_attr[0](edge_weight[sca_mask, 1])
        if gro_mask.any():
            idx = 1 if len(self.float_edge_attr) > 1 else 0
            attr_embed[gro_mask] = self.float_edge_attr[idx](edge_weight[gro_mask, 2])

        edge_embed = self.edge_linear(torch.cat([type_embed, attr_embed], dim=-1))
        edge_embed = F.normalize(edge_embed, p=2, dim=-1)
        edge_embed = self.dropout(self.act(edge_embed))

        x_prop = x.clone()
        x_gro = x.clone()
        prop_edges = edge_index[:, prop_mask | sca_mask]
        prop_embed = edge_embed[prop_mask | sca_mask]
        gro_edges = edge_index[:, prop_mask | gro_mask]
        gro_embed = edge_embed[prop_mask | gro_mask]

        # Offset should jump by num_layers, because blocks are appended in
        # [prop_layer0, prop_layer1, ..., gro_layer0, gro_layer1, ...] order.
        offset = self.num_layers
        for layer in range(self.num_layers):
            x_prop = self.blocks[layer](x_prop, prop_edges, prop_embed)
            idx = layer + offset
            x_gro = self.blocks[idx](x_gro, gro_edges, gro_embed)

        out = torch.cat([x_prop, x_gro], dim=-1)
        out = self.out_linear(out)
        if self.out_bn is not None:
            out = self.out_bn(out)
        return self.dropout(self.act(out))
