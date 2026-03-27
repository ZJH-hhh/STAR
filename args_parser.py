import argparse


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
    parser.add_argument("--rel_top_k", default=None, type=int)
    parser.add_argument("--rel_edge_hidden_dim", default=100, type=int)
    parser.add_argument("--rel_dropout", default=0.1, type=float)
    parser.add_argument("--rel_pre_dropout", default=0.1, type=float)
    parser.add_argument("--rel_nan_w", default=1., type=float)
    parser.add_argument("--rel_nan_type", default='nan', type=str, choices=['nan', '0', '1'])
    parser.add_argument("--rel_batch_norm", default=1, type=int)
    parser.add_argument("--rel_edge_type", default=1, type=int)
    # relation backend (Stage B)
    parser.add_argument("--relation_backend", default='baseline', type=str,
                        choices=['baseline', 'krgts'],
                        help="baseline: TaskRelationNet; krgts: HMRGNN-based relation net")
    parser.add_argument("--rel_hidden_dim", default=128, type=int,
                        help="Hidden dim used by the KRGTS relation backend")
    parser.add_argument("--rel_norm", default=1, type=int,
                        help="Enable L2 normalization before classification in HMRGNN backend")
    parser.add_argument("--num_relation_attr", default=3, type=int,
                        help="Number of relation attribute bins (prop/sca/gro) for KRGTS backend")
    # knowledge-edge fusion controls
    parser.add_argument("--use_knowledge_edges", default='none', type=str,
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
    parser.add_argument("--meta_lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--second_order", default=1, type=int)
    parser.add_argument("--inner_update_step", default=1, type=int)
    parser.add_argument("--inner_tasks", default=10, type=int)

    # few-shot
    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--n_support", default=10, type=int)
    parser.add_argument("--n_query", default=16, type=int)
    parser.add_argument("--n_test_tasks", default=200, type=int)
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--train_auxi_task_num", default=None, type=int)
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
    parser.add_argument("--use_semantic_sampler", default=0, type=int,
                        help="Enable semantic-guided auxiliary task sampling")
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
    parser.add_argument("--use_ltafm", default=0, type=int,
                        help="Enable LLM-driven task-adaptive feature modulation (FiLM)")
    parser.add_argument("--ltafm_semantic_dim", default=768, type=int,
                        help="Dimension of LLM semantic embeddings (default: SciBERT 768)")
    parser.add_argument("--ltafm_hidden_dim", default=256, type=int,
                        help="Hidden dimension for L-TAFM MLP")
    parser.add_argument("--ltafm_gamma_init", default=0.0, type=float,
                        help="Initialization bias for gamma (0.0 -> initial gamma ≈ 1)")
    parser.add_argument("--ltafm_diversity_weight", default=0.01, type=float,
                        help="Weight for gamma diversity regularization loss")

    args = parser.parse_args()

    if args.rel_top_k is None:
        # 限制 rel_top_k 的最大值为 5，避免引入过多噪声
        args.rel_top_k =args.n_support - 1 if args.n_support > 1 else 1
    
    # Semantic Sampler flags
    args.use_semantic_sampler = bool(args.use_semantic_sampler)
    if args.use_semantic_sampler and not args.semantic_embedding_path:
        # Auto-set embedding path based on dataset (with subdirectory structure)
        args.semantic_embedding_path = f"data/embeddings/{args.dataset}/{args.dataset}_property_embeddings.pt"
    
    # L-TAFM flags
    args.use_ltafm = bool(args.use_ltafm)
    if args.use_ltafm and not args.semantic_embedding_path:
        # Auto-set embedding path (L-TAFM requires semantic embeddings)
        args.semantic_embedding_path = f"data/embeddings/{args.dataset}/{args.dataset}_property_embeddings.pt"
    
    return args
