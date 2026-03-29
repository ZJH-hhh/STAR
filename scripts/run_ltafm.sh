#!/bin/bash
# =============================================================================
# L-TAFM (Task-Adaptive Feature Modulation) Experiment on SIDER
# =============================================================================
# 本脚本演示如何启用三个创新点的协同工作：
# - Innovation 1: KRGTS 关系网络（骨架/官能团知识边）
# - Innovation 2: LLM 语义引导的辅助任务采样 (Semantic Sampler)
# - Innovation 3: L-TAFM 任务自适应特征调制（FiLM）
# =============================================================================


#SBATCH --job-name=tox2110_lr0.05           # 任务名称
#SBATCH --partition=gpu-a100      # 请求的分区
#SBATCH --nodes=1                         # 请求一个节点
#SBATCH --ntasks=1                        # 请求一个任务
#SBATCH --cpus-per-task=20                # 每个任务使用一个 CPU
#SBATCH --gres=gpu:1


source /home/zjhzjh/miniconda3/etc/profile.d/conda.sh
conda activate pyg

# =============================================================================
# Global Settings
# =============================================================================
GPU=0             # 默认使用 GPU 2，可通过命令行参数覆盖
DATASET="tox21"
N_SUPPORT=10            # Support Set 大小
N_QUERY=8              # Query Set 大小
EPISODE=2000            # 训练轮次
EVAL_STEP=100           # 评估间隔
TEST_BATCH_SIZE=128

# =============================================================================
# Common Configuration (Architecture, Optimization, Knowledge)
# =============================================================================
# 这里包含了基础 GNN 设置、关系网络参数、知识蒸馏参数以及优化器设置
COMMON_ARGS="--dataset ${DATASET} \
    --gpu ${GPU} \
    --random_seed 0 \
    --n_support ${N_SUPPORT} \
    --n_query ${N_QUERY} \
    --episode ${EPISODE} \
    --eval_step ${EVAL_STEP} \
    --test_batch_size ${TEST_BATCH_SIZE} \
    --mol_num_layer 5 \
    --emb_dim 300 \
    --mol_dropout 0.1 \
    --mol_graph_pooling mean \
    --mol_gnn_type gin \
    --relation_backend krgts \
    --rel_layer 2 \
    --rel_hidden_dim 128 \
    --rel_dropout 0.1 \
    --rel_pre_dropout 0.1 \
    --rel_norm 1 \
    --rel_top_k 1 \
    --num_relation_attr 3 \
    --use_knowledge_edges both \
    --know_weight 0.4 \
    --know_temp 1.5 \
    --know_warmup 100 \
    --know_ramp 100 \
    --know_hold 0 \
    --know_decay 0 \
    --inner_lr 0.05 \
    --meta_lr 2e-4 \
    --weight_decay 5e-5 \
    --inner_update_step 3 \
    --inner_tasks 10 \
    --pool_num 20 \
    --task_lr 5e-4 \
    --task_hid_dim 10 \
    --nce_t 0.08 \
    --contr_w 0.05"


# =============================================================================
# Experiment: L-TAFM + Semantic Sampler + KRGTS
# =============================================================================
# 参数详解：
# --use_semantic_sampler: 启用基于 LLM 语义的辅助任务采样
# --use_ltafm 1: 启用 L-TAFM 特征调制模块
# --ltafm_hidden_dim 256: FiLM 生成器 MLP 的隐藏层维度（语义维度自动从嵌入文件读取）
# --ltafm_gamma_init 0.0: gamma 初始化偏置（0.0 对应初始 scaling ≈ 1，保持稳定）
# --ltafm_diversity_weight 0.01: 正则化项，鼓励不同任务生成多样的调制参数

echo "=========================================="
echo "Running L-TAFM Experiment on ${DATASET}"
echo "Innovation: KRGTS + Semantic Sampler + FiLM Modulation"
echo "=========================================="


python run.py ${COMMON_ARGS} \
    --exp_name ltafm \
    --exp_id ${DATASET}_${N_SUPPORT}shot_${inner_lr} \
    --use_semantic_sampler 1 \
    --semantic_temperature 0.1 \
    --semantic_similarity_weight 0.5 \
    --semantic_adjust_mode additive \
    --semantic_reward_bonus 0.1 \
    --semantic_warmup 0 \
    --use_ltafm 1 \
    --ltafm_hidden_dim 512 \
    --ltafm_gamma_init 0.0 \
    --ltafm_diversity_weight 0.005

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "L-TAFM Experiment completed!"
echo "=========================================="
echo ""
echo "Expected outcomes:"
echo "  - KRGTS 提供结构化的分子知识支持"
echo "  - Semantic Sampler 过滤辅助任务中的噪声，提高任务相关性"
echo "  - L-TAFM 根据任务语义动态调制特征 (Feature Modulation)"