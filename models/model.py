from copy import deepcopy

import numpy as np
from semantic_utils import load_task_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_utils import validate_semantic_embedding_metadata
from .base_encoder import GNN_Encoder
from .krgts_relation import KRGTSRelation
from .ltafm import AssayIDEmbedding, TaskAdaptiveModulator
from .relation import TaskRelationNet


class GSMeta(nn.Module):
    def __init__(self, task_num, train_task_num, args, test_task_ids=None):
        super(GSMeta, self).__init__()
        self.mol_encoder = GNN_Encoder(
            num_layer=args.mol_num_layer,
            emb_dim=args.emb_dim,
            JK=args.JK,
            drop_ratio=args.mol_dropout,
            graph_pooling=args.mol_graph_pooling,
            gnn_type=args.mol_gnn_type,
            batch_norm=args.mol_batch_norm,
            load_path=args.mol_pretrain_load_path,
        )

        # L-TAFM: Task-Adaptive Feature Modulation
        self.use_ltafm = getattr(args, "use_ltafm", False)
        self.conditioning = getattr(args, "conditioning", "text")
        self.id_seed = args.random_seed
        if self.conditioning == "support_adapted_id" and not self.use_ltafm:
            raise ValueError("Support-adapted ID conditioning requires L-TAFM")
        if self.use_ltafm:
            import os

            hidden_dim = getattr(args, "ltafm_hidden_dim", 256)
            gamma_init = getattr(args, "ltafm_gamma_init", 0.0)
            diversity_weight = getattr(args, "ltafm_diversity_weight", 0.01)
            dropout = getattr(args, "mol_dropout", 0.1)

            semantic_path = getattr(args, "semantic_embedding_path", "")
            if self.conditioning == "support_adapted_id":
                semantic_dim = args.id_dim
                if not isinstance(semantic_dim, int) or semantic_dim < 1:
                    raise ValueError("Use the matching Full embedding width for id_dim")
                self.task_ids = AssayIDEmbedding(task_num, semantic_dim)
            elif semantic_path and os.path.exists(semantic_path):
                emb_data = load_task_embeddings(semantic_path)
                loaded_encoder = validate_semantic_embedding_metadata(
                    data=emb_data,
                    expected_encoder=getattr(args, "semantic_encoder", None),
                    embedding_path=semantic_path,
                    consumer="L-TAFM",
                )
                embedding_matrix = emb_data["embedding_matrix"]
                if embedding_matrix.shape[0] != task_num:
                    raise ValueError("Assay embedding count differs from dataset task count")
                self.register_buffer("semantic_embeddings", embedding_matrix)
                semantic_dim = embedding_matrix.shape[1]
                if loaded_encoder is not None:
                    print(f"[L-TAFM] Embedding encoder: {loaded_encoder}")
            else:
                raise FileNotFoundError(f"Missing assay embeddings: {semantic_path}")


            self.ltafm = TaskAdaptiveModulator(
                feature_dim=args.emb_dim,
                semantic_dim=semantic_dim,
                hidden_dim=hidden_dim,
                gamma_init=gamma_init,
                diversity_weight=diversity_weight,
                dropout=dropout,
            )
        else:
            self.ltafm = None

        self.relation_backend = getattr(args, "relation_backend", "baseline")
        if self.relation_backend == "krgts":
            self.relation_net = KRGTSRelation(
                in_dim=args.emb_dim,
                total_tasks=task_num,
                train_tasks=train_task_num,
                test_task_ids=test_task_ids,
                args=args,
            )
        else:
            self.relation_net = TaskRelationNet(
                in_dim=args.emb_dim,
                num_layer=args.rel_layer,
                edge_n_layer=args.rel_edge_n_layer,
                edge_hidden_dim=args.rel_edge_hidden_dim,
                total_tasks=task_num,
                train_tasks=train_task_num,
                test_task_ids=test_task_ids,
                batch_norm=args.rel_batch_norm,
                top_k=args.rel_top_k,
                dropout=args.rel_dropout,
                pre_dropout=args.rel_pre_dropout,
                nan_w=args.rel_nan_w,
                nan_type=args.rel_nan_type,
                edge_type=args.rel_edge_type,
            )

    def conditioning_vectors(self, task_ids):
        if self.conditioning == "support_adapted_id":
            return self.task_ids(task_ids)
        return self.semantic_embeddings[task_ids]

    def initialize_held_out_id(self, task_id):
        """Start every repeat for this assay from the same independent, label-free draw."""
        if self.conditioning == "support_adapted_id":
            self.task_ids.reset_held_out(task_id, self.id_seed)

    def encode_mol(self, data):
        return self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)

    def forward(
        self,
        s_data,
        q_data,
        s_y,
        q_y,
        sampled_task,
        know_w: float = 0.0,
        know_temp: float = 1.0,
    ):
        s_feat, q_feat = self.encode_mol(s_data), self.encode_mol(q_data)

        diversity_loss = None
        if self.use_ltafm and self.ltafm is not None:
            target_task_id = sampled_task[0] if sampled_task.dim() > 0 else sampled_task
            task_semantic = self.conditioning_vectors(target_task_id)
            # The same assay-specific FiLM parameters apply to support and query molecules.
            combined, _ = self.ltafm(torch.cat([s_feat, q_feat]), task_semantic,
                                     compute_diversity_loss=False)
            s_feat, q_feat = combined.split([s_feat.size(0), q_feat.size(0)])

        s_feat_input = s_feat.repeat(q_feat.shape[0], 1, 1)
        q_feat_input = q_feat.unsqueeze(1)
        sample_feat = torch.cat([s_feat_input, q_feat_input], dim=1)
        know_adj = getattr(q_data, "know_adj", None)

        if self.relation_backend == "krgts":
            know_components = {
                "sca": getattr(q_data, "know_sca", None),
                "gro": getattr(q_data, "know_gro", None),
            }
            result = self.relation_net.forward_inductive(
                sample_feat,
                sampled_task,
                s_y,
                q_y,
                know_components=know_components,
            )
            s_logit, q_logit, s_label, q_label, graph_f = result
        else:
            s_logit, q_logit, s_label, q_label, graph_f = self.relation_net.forward_inductive(
                sample_feat,
                sampled_task,
                s_y,
                q_y,
                know_adj=know_adj,
                know_w=know_w,
                know_temp=know_temp,
            )

        return s_logit, q_logit, s_label, q_label, graph_f, s_feat, q_feat, diversity_loss
