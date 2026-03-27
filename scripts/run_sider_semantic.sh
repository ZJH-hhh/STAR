#!/bin/bash
# =============================================================================
# Semantic-Guided Task Sampling Experiments on SIDER
# =============================================================================
# 测试基于属性语义相似度的辅助任务采样策略
# 核心思想：选择与目标任务语义相关的辅助任务，避免噪声干扰
# =============================================================================


source /nfs/home/fragenmgr/miniconda3/etc/profile.d/conda.sh
conda activate pyg

GPU=${1:-3}
DATASET="tox21"
N_SUPPORT=1
EPISODE=2000
EVAL_STEP=100

# Common args
COMMON_ARGS="--dataset ${DATASET} \
    --n_support ${N_SUPPORT} \
    --episode ${EPISODE} \
    --eval_step ${EVAL_STEP} \
    --exp_name semantic_moderate \
    --gpu ${GPU} \
    --relation_backend krgts"
    # --mol_pretrain_load_path pretrained/supervised_contextpred.pth

# =============================================================================
# Experiment 3: Semantic Sampling - Moderate
# =============================================================================
echo "=========================================="
echo "Experiment 3: Semantic Sampling - Moderate"
echo "=========================================="
python run.py ${COMMON_ARGS} \
    --exp_id tox21_1shot \
    --use_semantic_sampler 1 \
    --semantic_temperature 0.1 \
    --semantic_similarity_weight 0.5 \
    --semantic_reward_bonus 0.1 \
    --semantic_warmup 50 \

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Expected outcomes:"
echo "  - Semantic guidance should help select more relevant auxiliary tasks"
echo "  - Conservative settings for stability, aggressive for exploration"
echo "  - Combination with knowledge edges may provide complementary benefits"

