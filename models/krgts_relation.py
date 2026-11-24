from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .graph_connector import InductiveGraphConnector
from .krgts_hgnn import HMRGNN


class KRGTSRelation(nn.Module):
    """Stage-B relation backend built on the HMRGNN encoder."""

    def __init__(self, in_dim, total_tasks, train_tasks, args, test_task_ids=None):
        super().__init__()
        self.args = args
        self.hidden_dim = args.rel_hidden_dim
        self.input_dim = in_dim
        self.task_emb = nn.Embedding(total_tasks, in_dim)
        held_out = slice(train_tasks, None) if test_task_ids is None else test_task_ids
        with torch.no_grad():
            self.task_emb.weight[held_out, :] = 0
        self.pre_dropout = nn.Dropout(args.rel_pre_dropout) if args.rel_pre_dropout > 0 else None
        self.input_proj = nn.Linear(in_dim, self.hidden_dim)
        self.graph_encoder = HMRGNN(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=args.rel_layer,
            num_edge_type=3,
            num_edge_attr_int=1,
            num_edge_attr_float=2,
            num_relation_attr=getattr(args, "num_relation_attr", 3),
            batch_norm=bool(args.rel_batch_norm),
            dropout=args.rel_dropout,
            norm=bool(args.rel_norm),
        )
        
        # Standard classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _prep_inputs(self, sample_emb, task_id):
        n_q = sample_emb.shape[0]
        tasks = self.task_emb(task_id)
        sample_emb = self.input_proj(sample_emb)
        task_emb = self.input_proj(tasks)
        if self.pre_dropout:
            sample_emb = self.pre_dropout(sample_emb)
            task_emb = self.pre_dropout(task_emb)
        task_emb = task_emb.unsqueeze(0).expand(n_q, -1, -1)
        input_nodes = torch.cat([task_emb, sample_emb], dim=1)
        flat_nodes = input_nodes.reshape(-1, input_nodes.shape[-1])
        return input_nodes, flat_nodes

    def _stack_edges(
        self,
        edge_ls,
        edge_type_ls,
        edge_w_ls,
        know_comps: Dict[str, torch.Tensor],
        n_task,
        device,
    ):
        n_q = len(edge_ls)
        n_s = know_comps["sca"].shape[1] - 1
        nodes_per_q = n_task + n_s + 1

        all_edges = []
        all_types = []
        all_weights = []

        def _append(edges, edge_types, prop_vals, sca_vals, gro_vals):
            all_edges.append(edges)
            all_types.append(edge_types)
            w = torch.zeros(edge_types.shape[0], 3, device=edges.device, dtype=torch.float32)
            w[:, 0] = prop_vals.float()
            if sca_vals is not None:
                w[:, 1] = sca_vals
            if gro_vals is not None:
                w[:, 2] = gro_vals
            all_weights.append(w)

        for qi in range(n_q):
            offset = qi * nodes_per_q
            # task <-> molecule edges (prop)
            edges_prop = edge_ls[qi].to(device=device).transpose(0, 1).to(torch.long) + offset
            types_prop = torch.zeros(edges_prop.shape[0], device=device, dtype=torch.long)
            prop_vals = edge_type_ls[qi].to(device=device).view(-1) - 1
            _append(edges_prop, types_prop, prop_vals, None, None)

            # scaffold knowledge edges
            sca_adj = know_comps["sca"][qi].to(device)
            sca_indices = torch.nonzero(sca_adj > 0, as_tuple=False)
            if sca_indices.numel() > 0:
                sca_edges = sca_indices + torch.tensor([offset + n_task, offset + n_task], device=device)
                sca_types = torch.ones(sca_edges.shape[0], device=device, dtype=torch.long)
                sca_vals = sca_adj[sca_indices[:, 0], sca_indices[:, 1]]
                _append(sca_edges, sca_types, torch.zeros_like(sca_vals), sca_vals, None)

            # group knowledge edges
            gro_adj = know_comps["gro"][qi].to(device)
            gro_indices = torch.nonzero(gro_adj > 0, as_tuple=False)
            if gro_indices.numel() > 0:
                gro_edges = gro_indices + torch.tensor([offset + n_task, offset + n_task], device=device)
                gro_types = torch.full((gro_edges.shape[0],), 2, device=device, dtype=torch.long)
                gro_vals = gro_adj[gro_indices[:, 0], gro_indices[:, 1]]
                _append(gro_edges, gro_types, torch.zeros_like(gro_vals), None, gro_vals)

        if not all_edges:
            return (
                torch.zeros(2, 0, device=device, dtype=torch.long),
                torch.zeros(0, device=device, dtype=torch.long),
                torch.zeros(0, 3, device=device),
            )

        edge_index = torch.cat(all_edges, dim=0).t().contiguous()
        edge_types = torch.cat(all_types, dim=0)
        edge_weights = torch.cat(all_weights, dim=0)
        return edge_index, edge_types, edge_weights

    def forward_inductive(
        self,
        sample_emb,
        task_id,
        support_y,
        query_y,
        know_components: Optional[Dict[str, torch.Tensor]] = None,
        **unused,
    ):
        device = sample_emb.device
        if know_components is None or know_components.get('sca') is None:
            zeros = sample_emb.new_zeros(sample_emb.shape[0], sample_emb.shape[1], sample_emb.shape[1])
            know_components = {"sca": zeros, "gro": zeros}
        else:
            source = know_components
            know_components = {
                k: (v.to(device) if v is not None else torch.zeros_like(source['sca']))
                for k, v in source.items()
            }

        tgt_s_y, tgt_q_y, tgt_s_idx, tgt_q_idx, edge_ls, edge_type_ls, edge_w_ls = InductiveGraphConnector.connect_task_and_sample(
            support_y, query_y, self.args.rel_nan_w, self.args.rel_nan_type
        )

        node_tensor, flat_nodes = self._prep_inputs(sample_emb, task_id)
        edge_index, edge_types, edge_weights = self._stack_edges(
            edge_ls, edge_type_ls, edge_w_ls, know_components, n_task=len(task_id), device=sample_emb.device
        )
        type_tensor = torch.tensor([0, 1, 2], device=edge_types.device, dtype=torch.long)
        enc = self.graph_encoder(flat_nodes, edge_index, type_tensor, edge_types, edge_weights)
        enc = enc.view(node_tensor.shape)

        n_q, n_task, n_s = sample_emb.shape[0], support_y.shape[1], support_y.shape[0]
        support_sample = torch.cat(
            [enc[:, tgt_s_idx[:, 0], :], enc[:, tgt_s_idx[:, 1], :]],
            dim=-1,
        )
        query_sample = torch.cat(
            [enc[:, tgt_q_idx[:, 0], :], enc[:, tgt_q_idx[:, 1], :]],
            dim=-1,
        )
        if not self.args.rel_norm:
            pass
        else:
            support_sample = F.normalize(support_sample, p=2, dim=-1)
            query_sample = F.normalize(query_sample, p=2, dim=-1)

        # Compute logits
        support_logit = self.classifier(support_sample.contiguous().view(n_q * n_s, -1))
        support_logit = support_logit.view(n_q, n_s)
        query_logit = self.classifier(query_sample.contiguous().view(n_q, -1))
        
        summary = enc.mean(0)
        graph_emb = summary[0] + torch.sigmoid(summary[1:].mean(0))
        
        return support_logit, query_logit, tgt_s_y.repeat(n_q, 1, 1).view(n_q, n_s), tgt_q_y, graph_emb
