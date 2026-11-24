import argparse
import csv
import os

from semantic_utils import default_semantic_embedding_path, normalize_semantic_encoder


def args_parser():
    parser = argparse.ArgumentParser()
    # exp
    parser.add_argument("--exp_name", default="run", type=str,
                        help="Experiment name")
    parser.add_argument("--dump_path", default="logs/", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--random_seed", default=0, type=int)


    # dataset
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--dataset", default='sider', type=str)
                        # choices=['sider', 'tox21', 'muv', 'toxcast'])
    parser.add_argument("--task_sampling", default='random', type=str,
                        choices=['random', 'scaffold'],
                        help="Episode sampling strategy for support/query selection")
    # mol encoder
    parser.add_argument("--mol_num_layer", default=5, type=int)
    parser.add_argument("--emb_dim", default=300, type=int)
    parser.add_argument("--JK", default='last', type=str)
    parser.add_argument("--mol_dropout", default=0.1, type=float)
    parser.add_argument("--mol_graph_pooling", default='mean', type=str)
    parser.add_argument("--mol_gnn_type", default='gin', type=str)
    parser.add_argument("--mol_batch_norm", default=1, type=int)
    parser.add_argument("--mol_pretrain_load_path", default=None)

    # relation net
    parser.add_argument("--rel_layer", default=2, type=int)
    parser.add_argument("--rel_edge_n_layer", default=2, type=int)
    parser.add_argument("--rel_top_k", default=1, type=int)
    parser.add_argument("--rel_edge_hidden_dim", default=100, type=int)
    parser.add_argument("--rel_dropout", default=0.1, type=float)
    parser.add_argument("--rel_pre_dropout", default=0.1, type=float)
    parser.add_argument("--rel_nan_w", default=0., type=float,
                        help="Weight of explicit missing-label edges. Zero uses observed labels only.")
    parser.add_argument("--rel_nan_type", default='nan', type=str, choices=['nan', '0', '1'])
    parser.add_argument("--rel_batch_norm", default=1, type=int)
    parser.add_argument("--rel_edge_type", default=1, type=int)
    # relation backend (Stage B)
    parser.add_argument("--relation_backend", default='krgts', type=str,
                        choices=['baseline', 'krgts'],
                        help="baseline: TaskRelationNet; krgts: HMRGNN-based relation net")
    parser.add_argument("--rel_hidden_dim", default=128, type=int,
                        help="Hidden dim used by the KRGTS relation backend")
    parser.add_argument("--rel_norm", default=1, type=int,
                        help="Enable L2 normalization before classification in HMRGNN backend")
    parser.add_argument("--num_relation_attr", default=3, type=int,
                        help="Number of relation attribute bins (prop/sca/gro) for KRGTS backend")
    # knowledge-edge fusion controls
    parser.add_argument("--use_knowledge_edges", default='both', type=str,
                        choices=['none', 'sca', 'gro', 'both'],
                        help="Fuse scaffold/group similarity edges into relation graph")
    parser.add_argument("--know_weight", default=0.4, type=float,
                        help="Max blending weight for knowledge adjacency in fusion (0..1)")
    parser.add_argument("--know_temp", default=1.5, type=float,
                        help="Temperature for knowledge adjacency row-wise softmax (>=1.0 is safer)")
    parser.add_argument("--know_warmup", default=100, type=int,
                        help="Episodes to keep knowledge weight at 0 during training warm-up")
    parser.add_argument("--know_ramp", default=100, type=int,
                        help="Episodes after warm-up to linearly ramp knowledge weight to know_weight")
    parser.add_argument("--know_hold", default=0, type=int,
                        help="Episodes to hold knowledge weight at peak after ramp")
    parser.add_argument("--know_decay", default=0, type=int,
                        help="Episodes to linearly decay knowledge weight back to 0 after hold")

    # maml
    parser.add_argument("--inner_lr", default=0.05, type=float)
    parser.add_argument("--meta_lr", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--second_order", default=1, type=int)
    parser.add_argument("--inner_update_step", default=3, type=int)
    parser.add_argument("--inner_tasks", default=10, type=int)

    # few-shot
    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--n_support", default=10, type=int)
    parser.add_argument("--n_query", default=8, type=int)
    parser.add_argument("--n_test_tasks", default=200, type=int)
    parser.add_argument("--revision_root", default=".", help="Extracted revision data directory")
    parser.add_argument("--eval_manifest", help="Override the archived JSONL support/query manifest")
    parser.add_argument("--fingerprint_mode", choices=["corrected", "legacy"], default="corrected")
    parser.add_argument("--mask_query_aux", action="store_true", help="Hide query auxiliary labels during evaluation")
    parser.add_argument("--eval_checkpoint", help="Evaluate a saved fixed-endpoint model without training")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Export compressed molecule predictions and empty-scaffold flags")
    parser.add_argument("--attribution_episodes",
                        help="Analyze selected query molecules using supplied attribution episodes and checkpoint")

    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--train_auxi_task_num", default=None, type=int,
                        help="Optional training draw count; by default use all other meta-training assays.")
    parser.add_argument("--test_auxi_task_num", default=None, type=int)

    # contrastive
    parser.add_argument("--nce_t", default=0.08, type=float)
    parser.add_argument("--contr_w", default=0.05, type=float)
    # selector
    parser.add_argument("--pool_num", default=20, type=int)
    parser.add_argument("--task_lr", default=5e-4, type=float)
    parser.add_argument("--task_hid_dim", default=10, type=int)
    parser.add_argument("--task_t", default=1, type=float)

    # Semantic-Guided Task Sampling (LLM-based Property Descriptions)
    parser.add_argument("--use_semantic_sampler", default=1, type=int,
                        help="Enable semantic-guided auxiliary task sampling")
    parser.add_argument("--conditioning", choices=["text", "support_adapted_id"], default="text")
    parser.add_argument("--id_dim", type=int, help="ID width (default: Full width from S28)")
    parser.add_argument("--text_fit", choices=["all_corpus", "train_only"], default="all_corpus")
    parser.add_argument("--text_variant", choices=["full", "name", "shuffled", "random"], default="full")
    parser.add_argument("--semantic_encoder", default="tfidf", type=str,
                        choices=["tfidf", "bert", "sbert", "sentence-transformers"],
                        help="Text encoder used to build property embeddings")
    parser.add_argument("--semantic_embedding_path", default="", type=str,
                        help="Path to precomputed property embeddings (auto-set if empty)")
    parser.add_argument("--semantic_temperature", default=0.1, type=float,
                        help="Temperature for semantic similarity softmax (lower = sharper)")
    parser.add_argument("--semantic_similarity_weight", default=0.5, type=float,
                        help="Weight for semantic similarity in sampling distribution")
    parser.add_argument("--semantic_min_similarity", default=-1.0, type=float,
                        help="Minimum similarity threshold (-1.0 = no threshold)")
    parser.add_argument("--semantic_adjust_mode", default='additive', type=str,
                        choices=['additive', 'multiplicative', 'gating'],
                        help="Mode for adjusting sampling logits with semantic similarity")
    parser.add_argument("--semantic_reward_bonus", default=0.1, type=float,
                        help="Bonus weight for semantic similarity in sampler reward")
    parser.add_argument("--semantic_warmup", default=0, type=int,
                        help="Episodes to keep semantic guidance disabled (warm-up)")
    
    # L-TAFM: Task-Adaptive Feature Modulation (Innovation 3)
    parser.add_argument("--use_ltafm", default=1, type=int,
                        help="Enable LLM-driven task-adaptive feature modulation (FiLM)")
    parser.add_argument("--ltafm_semantic_dim", default=768, type=int,
                        help="Dimension of LLM semantic embeddings (default: SciBERT 768)")
    parser.add_argument("--ltafm_hidden_dim", default=512, type=int,
                        help="Hidden dimension for L-TAFM MLP")
    parser.add_argument("--ltafm_gamma_init", default=0.0, type=float,
                        help="Initialization bias for gamma (0.0 -> initial gamma ≈ 1)")
    parser.add_argument("--ltafm_diversity_weight", default=0.005, type=float,
                        help="Weight for gamma diversity regularization loss")

    args = parser.parse_args()

    if args.rel_top_k is None:
        # 限制 rel_top_k 的最大值为 5，避免引入过多噪声
        args.rel_top_k =args.n_support - 1 if args.n_support > 1 else 1
    
    # Semantic Sampler flags
    args.semantic_encoder = normalize_semantic_encoder(args.semantic_encoder)
    args.use_semantic_sampler = bool(args.use_semantic_sampler)
    args.use_ltafm = bool(args.use_ltafm)
    if args.attribution_episodes and not args.eval_checkpoint:
        parser.error("--attribution_episodes requires --eval_checkpoint")
    if args.conditioning == "support_adapted_id":
        if not args.use_ltafm or args.semantic_embedding_path:
            parser.error("support_adapted_id uses L-TAFM and takes no text embedding file")
        args.use_semantic_sampler = False
        if args.id_dim is None:
            table = os.path.join(args.revision_root, "Supplementary_Effective_Embedding_Dimensions.csv")
            if not os.path.isfile(table):
                parser.error("Provide the S28 table under --revision_root or set --id_dim explicitly")
            with open(table, encoding="utf-8-sig", newline="") as stream:
                widths = [int(row[args.text_fit + "_dimension"]) for row in csv.DictReader(stream)
                          if row["dataset"] == args.dataset]
            if len(widths) != 1:
                parser.error("Expected one S28 row for " + args.dataset)
            args.id_dim = widths[0]
        if args.id_dim <= 0:
            parser.error("--id_dim must be positive")
    elif (args.use_semantic_sampler or args.use_ltafm) and not args.semantic_embedding_path:
        if args.semantic_encoder == "tfidf":
            variant = args.text_variant
            if variant in ("random", "shuffled"):
                variant += f"_seed{args.random_seed}"
            directory = os.path.join(args.revision_root, "text_embeddings", args.dataset)
            if args.text_fit == "all_corpus":
                args.semantic_embedding_path = os.path.join(directory, "all_corpus", variant + ".npz")
            else:
                args.semantic_embedding_path = os.path.join(directory, variant + ".pt")
        else:
            args.semantic_embedding_path = default_semantic_embedding_path(args.dataset, args.semantic_encoder)
    if args.semantic_embedding_path:
        args.semantic_embedding_path = os.path.abspath(args.semantic_embedding_path)
    if not args.eval_manifest:
        args.eval_manifest = os.path.join(args.revision_root, "eval_manifests", args.dataset,
                                          f"{args.n_support}shot", f"seed_{args.random_seed}.jsonl")
    return args
